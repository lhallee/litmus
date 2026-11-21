//! Readings as CSV rows and JSON objects, one per file or text, in the same columns for the command line and the app.

use litmus_engine::{Chance, DocumentReading, Origin};

use crate::counts::TextCounts;

/// One scored input: its name (a file path or the app's title for a pasted text), the SHA-256 of its bytes, whether
/// the reading came from the cache, the reading or the error, the counts of the text that was read (JSON only), and
/// the verdict's chance of AI (`DocumentReading::chance`; the last three CSV columns, empty without it).
pub struct Row<'a> {
    pub name: &'a str,
    pub hash: &'a str,
    pub cached: bool,
    pub outcome: Result<&'a DocumentReading, &'a str>,
    pub counts: Option<&'a TextCounts>,
    pub chance: Option<Chance>,
}

/// The CSV header; `csv_line` writes the matching rows. `p_not_human` and its interval are calibrated on the
/// reference texts as they are mixed; `chance_of_ai` and its interval restate them at even odds, the number the app
/// and the command line show. Columns are added at the end, so a reader by position keeps its columns.
pub const CSV_HEADER: &str = "file,sha256,verdict,p_not_human,interval_lo,interval_hi,set_0.05,set_0.1,p_human,p_ai,p_mixed,three_way_human,three_way_ai,three_way_mixed,closeness_p,out_of_reference,strip_position,band_lo,band_hi,tokens,chunks,disagreement,mixed_document,map_lowest,map_highest,error,chance_of_ai,chance_lo,chance_hi";

/// A conformal set in words, as `{human, AI}`.
pub fn set_words(set: Option<&Vec<Origin>>) -> String {
    let names: Vec<&str> = set.map(Vec::as_slice).unwrap_or(&[]).iter().map(|o| if *o == Origin::Ai { "AI" } else { o.name() }).collect();
    format!("{{{}}}", names.join(", "))
}

/// An integer with thousands separated by commas.
pub fn thousands(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// A CSV field, quoted when it holds a comma, a quote or a line break.
pub fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) { format!("\"{}\"", value.replace('"', "\"\"")) } else { value.to_owned() }
}

/// The lowest and highest heat-map position of a reading.
pub fn map_range(document: &DocumentReading) -> (f64, f64) {
    document.cells.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), cell| (low.min(cell.position), high.max(cell.position)))
}

/// One CSV row (no line break) in the columns of `CSV_HEADER`.
pub fn csv_line(row: &Row) -> String {
    match row.outcome {
        Err(error) => format!("{},{},,,,,,,,,,,,,,,,,,,,,,,,{},,,", csv_field(row.name), row.hash, csv_field(error)),
        Ok(document) => {
            let v = &document.verdict;
            let set = |level: f64| set_words(v.set(level)).trim_matches(['{', '}']).replace(", ", " ");
            let (low, high) = map_range(document);
            let chance = row.chance.map_or_else(|| ",,".to_owned(), |c| format!("{},{},{}", c.value, c.interval[0], c.interval[1]));
            format!(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},,{chance}",
                csv_field(row.name),
                row.hash,
                document.label().word(),
                v.probability,
                v.interval[0],
                v.interval[1],
                set(0.05),
                set(0.1),
                v.p_values[0],
                v.p_values[1],
                v.p_values[2],
                v.three_way[0],
                v.three_way[1],
                v.three_way[2],
                v.closeness_p,
                v.out_of_reference,
                v.strip.position,
                v.strip.band_lo,
                v.strip.band_hi,
                document.n_tokens,
                v.chunks,
                v.disagreement,
                v.mixed_document,
                low,
                high,
            )
        }
    }
}

/// A header and one line per row, each ending in a line break.
pub fn csv(rows: &[Row]) -> String {
    let mut out = String::with_capacity(256 * (rows.len() + 1));
    out.push_str(CSV_HEADER);
    out.push('\n');
    for row in rows {
        out.push_str(&csv_line(row));
        out.push('\n');
    }
    out
}

/// One JSON object: the file, its hash, whether it was cached, the verdict in one word, every field of the reading
/// and, when the row has them, the text's `counts` (words, characters, characters without spaces, tokens) and the
/// verdict's `chance` of AI (`value`, `interval`, `reference_share`), or the error.
pub fn json_value(row: &Row) -> serde_json::Value {
    match row.outcome {
        Ok(document) => {
            let mut value = serde_json::json!({"file": row.name, "sha256": row.hash, "cached": row.cached,
                "label": document.label().word(), "reading": document});
            if let Some(counts) = row.counts {
                value["counts"] = counts.json();
            }
            if let Some(chance) = row.chance {
                value["chance"] = serde_json::json!(chance);
            }
            value
        }
        Err(error) => serde_json::json!({"file": row.name, "sha256": row.hash, "error": error}),
    }
}

/// JSON Lines: one object per row, each ending in a line break.
pub fn json_lines(rows: &[Row]) -> String {
    rows.iter().map(|row| format!("{}\n", json_value(row))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_and_fields() {
        assert_eq!(thousands(1_234_567), "1,234,567");
        assert_eq!(thousands(12), "12");
        assert_eq!(set_words(Some(&vec![Origin::Human, Origin::Ai])), "{human, AI}");
        assert_eq!(set_words(None), "{}");
        assert_eq!(csv_field("a,b \"c\""), "\"a,b \"\"c\"\"\"");
        assert_eq!(csv_field("plain"), "plain");
    }

    #[test]
    fn error_rows_keep_every_column() {
        let row = Row { name: "x.txt", hash: "ab", cached: false, outcome: Err("broken, badly"), counts: None, chance: None };
        assert_eq!(csv_line(&row).split(',').count(), CSV_HEADER.split(',').count() + 1); // the quoted comma
        assert_eq!(json_value(&row)["error"], "broken, badly");
        assert!(csv(&[row]).starts_with(CSV_HEADER));
    }
}

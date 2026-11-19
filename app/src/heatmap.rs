//! From a reading's cells to the text on screen. Every cell carries its byte range in the original text (the engine's
//! `Place`), and the cells tile the text, so the text is cut at cell boundaries into pages of a few cells each; each
//! page is laid out on its own and only the pages in view are painted, which keeps a 131,072-token document smooth.

use std::ops::Range;

use litmus_engine::{Cell, DocumentReading, WindowReading};

/// Cells per page of the heat map's text.
pub const CELLS_PER_PAGE: usize = 12;

/// A run of whole cells and the bytes of the text they cover.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Page {
    pub cells: Range<usize>,
    pub bytes: Range<usize>,
}

/// Check that a reading's places fit its text: chunks and cells tile it from byte 0 to its end, in order, on
/// character boundaries, and windows lie inside it. The app refuses to draw a reading that does not fit.
pub fn check_places(document: &DocumentReading, text: &str) -> Result<(), String> {
    let tiles = |name: &str, spans: Vec<(usize, usize)>| -> Result<(), String> {
        let mut at = 0;
        for (index, (start, end)) in spans.into_iter().enumerate() {
            if start != at || end < start || end > text.len() {
                return Err(format!("{name} {index} covers bytes {start} to {end}, expected to start at {at} within {} bytes", text.len()));
            }
            if !text.is_char_boundary(start) || !text.is_char_boundary(end) {
                return Err(format!("{name} {index} does not start and end on character boundaries"));
            }
            at = end;
        }
        if at != text.len() {
            return Err(format!("the {name}s end at byte {at}, the text at {}", text.len()));
        }
        Ok(())
    };
    tiles("cell", document.cells.iter().map(|c| (c.place.start_byte, c.place.end_byte)).collect())?;
    tiles("chunk", document.chunks.iter().map(|c| (c.place.start_byte, c.place.end_byte)).collect())?;
    if let Some(window) = document.windows.iter().find(|w| w.place.end_byte > text.len() || w.place.start_byte > w.place.end_byte) {
        return Err(format!("a window covers bytes {} to {}, past the text", window.place.start_byte, window.place.end_byte));
    }
    Ok(())
}

/// The pages of a reading's text, `per_page` cells each (the last may hold fewer).
pub fn pages(cells: &[Cell], per_page: usize) -> Vec<Page> {
    assert!(per_page > 0, "at least one cell per page");
    cells
        .chunks(per_page)
        .enumerate()
        .map(|(index, run)| {
            let first = index * per_page;
            Page { cells: first..first + run.len(), bytes: run[0].place.start_byte..run[run.len() - 1].place.end_byte }
        })
        .collect()
}

/// The cells of a page as byte ranges local to the page's text, with each cell's index.
pub fn sections(cells: &[Cell], page: &Page) -> Vec<(Range<usize>, usize)> {
    page.cells
        .clone()
        .map(|index| {
            let place = &cells[index].place;
            (place.start_byte - page.bytes.start..place.end_byte - page.bytes.start, index)
        })
        .collect()
}

/// The cell holding a byte of the text (the last cell for the text's end).
pub fn cell_at_byte(cells: &[Cell], byte: usize) -> Option<usize> {
    if cells.is_empty() {
        return None;
    }
    let index = cells.partition_point(|cell| cell.place.end_byte <= byte);
    Some(index.min(cells.len() - 1))
}

/// The byte offset of a character offset within `text`.
pub fn byte_of_char(text: &str, char_index: usize) -> usize {
    text.char_indices().nth(char_index).map_or(text.len(), |(byte, _)| byte)
}

/// The windows over a cell (overlapping it by token position, as the engine counts them), the one centered nearest
/// the cell first.
pub fn covering_windows(windows: &[WindowReading], cell: &Cell) -> Vec<usize> {
    let center = |a: usize, b: usize| (a + b) as f64 / 2.0;
    let cell_center = center(cell.place.start, cell.place.end);
    let mut covering: Vec<usize> = (0..windows.len()).filter(|&i| windows[i].place.start < cell.place.end && windows[i].place.end > cell.place.start).collect();
    covering.sort_by(|&a, &b| {
        let distance = |i: usize| (center(windows[i].place.start, windows[i].place.end) - cell_center).abs();
        distance(a).total_cmp(&distance(b)).then(a.cmp(&b))
    });
    covering
}

#[cfg(test)]
mod tests {
    use super::*;
    use litmus_engine::Place;

    fn cell(start: usize, end: usize, start_byte: usize, end_byte: usize) -> Cell {
        Cell {
            place: Place { start, end, start_byte, end_byte, start_char: 0, end_char: 0 },
            position: 0.5,
            spread: 0.0,
            band_lo: 0.0,
            band_hi: 1.0,
            windows: 1,
            unlike_reference: false,
        }
    }

    #[test]
    fn pages_cover_whole_cells() {
        let cells: Vec<Cell> = (0..5).map(|i| cell(i * 128, (i + 1) * 128, i * 10, (i + 1) * 10)).collect();
        let pages = pages(&cells, 2);
        assert_eq!(pages, vec![Page { cells: 0..2, bytes: 0..20 }, Page { cells: 2..4, bytes: 20..40 }, Page { cells: 4..5, bytes: 40..50 }]);
        assert_eq!(sections(&cells, &pages[1]), vec![(0..10, 2), (10..20, 3)]);
    }

    #[test]
    fn bytes_find_their_cell() {
        let cells: Vec<Cell> = (0..3).map(|i| cell(i * 128, (i + 1) * 128, i * 10, (i + 1) * 10)).collect();
        assert_eq!(cell_at_byte(&cells, 0), Some(0));
        assert_eq!(cell_at_byte(&cells, 9), Some(0));
        assert_eq!(cell_at_byte(&cells, 10), Some(1));
        assert_eq!(cell_at_byte(&cells, 29), Some(2));
        assert_eq!(cell_at_byte(&cells, 30), Some(2), "the end of the text is in the last cell");
        assert_eq!(cell_at_byte(&[], 3), None);
    }

    #[test]
    fn characters_map_to_bytes() {
        let text = "a\u{e9}b\u{1f600}c";
        assert_eq!(byte_of_char(text, 0), 0);
        assert_eq!(byte_of_char(text, 2), 3);
        assert_eq!(byte_of_char(text, 4), 8);
        assert_eq!(byte_of_char(text, 9), text.len());
    }
}

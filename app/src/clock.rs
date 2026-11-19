//! Wall-clock time for history entries, in milliseconds since the Unix epoch, written as UTC dates.

/// Milliseconds since the Unix epoch now.
#[cfg(not(target_arch = "wasm32"))]
pub fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64)
}

#[cfg(target_arch = "wasm32")]
pub fn now_ms() -> i64 {
    js_sys::Date::now() as i64
}

/// The civil date of a day count since 1970-01-01 (Howard Hinnant's `civil_from_days`).
fn civil(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(month <= 2), month, day)
}

/// `2026-11-17 18:05 UTC`.
pub fn format_utc(ms: i64) -> String {
    let seconds = ms.div_euclid(1000);
    let (year, month, day) = civil(seconds.div_euclid(86_400));
    let minute_of_day = seconds.rem_euclid(86_400) / 60;
    format!("{year:04}-{month:02}-{day:02} {:02}:{:02} UTC", minute_of_day / 60, minute_of_day % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_are_civil() {
        assert_eq!(format_utc(0), "1970-01-01 00:00 UTC");
        assert_eq!(format_utc(951_782_400_000), "2000-02-29 00:00 UTC");
        assert_eq!(format_utc(1_794_938_700_000), "2026-11-17 18:05 UTC");
        assert_eq!(format_utc(-60_000), "1969-12-31 23:59 UTC");
    }
}

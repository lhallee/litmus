//! Sorting the Batch and History tables by a clicked header: a click on a new column sorts by it (numbers high to
//! low first, words A to Z), a second click on the same column reverses it. Sorting is stable, numbers that are not
//! a number and rows without a value go last whichever the direction, and words compare without case.

use std::cmp::Ordering;

/// What a table is sorted by: a column and a direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sort<K> {
    pub key: K,
    pub descending: bool,
}

/// A row's value in the sorted column.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Number(f64),
    Text(String),
    /// No value (a waiting or failed row): always last.
    Missing,
}

impl<K: Copy + PartialEq> Sort<K> {
    /// The sort after a click on `key`'s header; `numeric` columns start high to low, word columns A to Z.
    pub fn clicked(current: Option<Self>, key: K, numeric: bool) -> Self {
        match current {
            Some(sort) if sort.key == key => Self { key, descending: !sort.descending },
            _ => Self { key, descending: numeric },
        }
    }
}

fn compare(a: &Value, b: &Value, descending: bool) -> Ordering {
    let flip = |o: Ordering| if descending { o.reverse() } else { o };
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => match (x.is_nan(), y.is_nan()) {
            (false, false) => flip(x.total_cmp(y)),
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
        },
        (Value::Text(x), Value::Text(y)) => flip(x.to_lowercase().cmp(&y.to_lowercase())),
        (Value::Missing, Value::Missing) => Ordering::Equal,
        (Value::Missing, _) => Ordering::Greater,
        (_, Value::Missing) => Ordering::Less,
        // Mixed kinds in one column: numbers before words, whatever the direction.
        (Value::Number(_), Value::Text(_)) => Ordering::Less,
        (Value::Text(_), Value::Number(_)) => Ordering::Greater,
    }
}

/// The order of `values`' indices under `descending` (stable: equal rows keep their order).
pub fn order(values: &[Value], descending: bool) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|&a, &b| compare(&values[a], &values[b], descending));
    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clicks_choose_and_reverse() {
        let first = Sort::clicked(None, 2, true);
        assert_eq!(first, Sort { key: 2, descending: true }, "numbers start high to low");
        assert_eq!(Sort::clicked(Some(first), 2, true), Sort { key: 2, descending: false }, "a second click reverses");
        assert_eq!(Sort::clicked(Some(first), 0, false), Sort { key: 0, descending: false }, "words start A to Z");
    }

    #[test]
    fn numbers_sort_with_gaps_last() {
        let values = vec![Value::Number(0.2), Value::Missing, Value::Number(f64::NAN), Value::Number(0.9), Value::Number(0.2)];
        assert_eq!(order(&values, false), [0, 4, 3, 2, 1], "ascending, stable for the ties, not-a-number then missing last");
        assert_eq!(order(&values, true), [3, 0, 4, 2, 1], "descending keeps the gaps last");
    }

    #[test]
    fn words_ignore_case() {
        let values = vec![Value::Text("beta".into()), Value::Text("Alpha".into()), Value::Text("alpha".into()), Value::Missing];
        assert_eq!(order(&values, false), [1, 2, 0, 3]);
        assert_eq!(order(&values, true), [0, 1, 2, 3], "ties keep their order in either direction");
    }
}

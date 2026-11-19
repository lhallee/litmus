//! Chunks, windows and cells (SPEC.md, Chunks; Windows and the heat map): a document of `n` tokens is read as
//! `C = ceil(n / C_max)` equal chunks, each a text of its own; inside a chunk, windows of `window_width` tokens every
//! `window_stride` are read from differences of running float64 sums, and the heat map has one cell per stride.

/// `C = ceil(n / chunk_tokens)` chunks; chunk `c` is `[floor(c n / C), floor((c + 1) n / C))`.
pub fn chunk_bounds(n_tokens: usize, chunk_tokens: usize) -> Vec<(usize, usize)> {
    assert!(n_tokens > 0 && chunk_tokens > 0, "a text of zero tokens has no chunks");
    let count = n_tokens.div_ceil(chunk_tokens);
    let edge = |index: usize| (index as u128 * n_tokens as u128 / count as u128) as usize;
    (0..count).map(|index| (edge(index), edge(index + 1))).collect()
}

/// The windows of a chunk of `length` tokens, chunk positions `[start, end)`: one window if the chunk is no wider
/// than a window, else one every `stride` and a last one ending at the chunk's end.
pub fn window_spans(length: usize, width: usize, stride: usize) -> Vec<(usize, usize)> {
    if length <= width {
        return vec![(0, length)];
    }
    let mut spans: Vec<(usize, usize)> = (0..=length - width).step_by(stride).map(|start| (start, start + width)).collect();
    if spans.last().is_some_and(|&(_, end)| end != length) {
        spans.push((length - width, length));
    }
    spans
}

/// The heat map's cells of a chunk: one per stride, the last possibly shorter.
pub fn cell_spans(length: usize, stride: usize) -> Vec<(usize, usize)> {
    (0..length).step_by(stride).map(|start| (start, (start + stride).min(length))).collect()
}

/// Running float64 sums of one chunk's token states, recorded at every position a window starts or ends; the pooled
/// vector of `[a, b)` is the difference of the sums recorded at `b` and at `a`.
#[derive(Clone, Debug)]
pub struct ChunkPool {
    width: usize,
    pub length: usize,
    pub count: usize,
    total: Vec<f64>,   // (d,)
    squares: Vec<f64>, // (d,)
    marks: Vec<usize>, // ascending positions with recorded sums, 0 first
    next: usize,       // index of the next mark to record
    recorded: Vec<f64>, // (marks, 2d): total then squares
}

impl ChunkPool {
    pub fn new(width: usize, length: usize, window_width: usize, window_stride: usize) -> Self {
        let mut marks: Vec<usize> = window_spans(length, window_width, window_stride).into_iter().flat_map(|(a, b)| [a, b]).collect();
        marks.push(0);
        marks.sort_unstable();
        marks.dedup();
        let mut recorded = Vec::with_capacity(marks.len() * 2 * width);
        recorded.resize(2 * width, 0.0);
        Self { width, length, count: 0, total: vec![0.0; width], squares: vec![0.0; width], marks, next: 1, recorded }
    }

    /// Bytes held at most, once every mark is recorded.
    pub fn bytes(&self) -> usize {
        8 * (2 * self.width + self.marks.len() * 2 * self.width) + 8 * self.marks.len()
    }

    /// Add the next token state, `(d,)`.
    #[inline]
    pub fn add(&mut self, row: &[f32]) {
        assert!(self.count < self.length, "a chunk pool takes only its chunk's tokens");
        for ((total, square), &value) in self.total.iter_mut().zip(self.squares.iter_mut()).zip(row) {
            let value = value as f64;
            *total += value;
            *square += value * value;
        }
        self.count += 1;
        if self.count == self.marks[self.next] {
            self.recorded.extend_from_slice(&self.total);
            self.recorded.extend_from_slice(&self.squares);
            self.next += 1;
        }
    }

    /// `float32(concat(mean, max(sumsq / n - mean^2, 0)))` of chunk positions `[start, end)`, both recorded: `(2d,)`.
    pub fn pooled(&self, start: usize, end: usize, out: &mut [f32]) {
        let d = self.width;
        let at = |position: usize| {
            let index = self.marks.binary_search(&position).expect("pooled spans start and end at recorded marks");
            assert!(index < self.next, "the span's end is not pooled yet");
            &self.recorded[index * 2 * d..(index + 1) * 2 * d]
        };
        let (first, last) = (at(start), at(end));
        let n = (end - start) as f64;
        assert!(end > start, "an empty span has no statistics");
        for i in 0..d {
            let mean = (last[i] - first[i]) / n;
            let variance = ((last[d + i] - first[d + i]) / n - mean * mean).max(0.0);
            out[i] = mean as f32;
            out[d + i] = variance as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_tile_the_document() {
        assert_eq!(chunk_bounds(5, 8), vec![(0, 5)]);
        assert_eq!(chunk_bounds(8192, 8192), vec![(0, 8192)]);
        assert_eq!(chunk_bounds(8193, 8192), vec![(0, 4096), (4096, 8193)]);
        let bounds = chunk_bounds(131_072, 8192);
        assert_eq!(bounds.len(), 16);
        assert!(bounds.iter().all(|&(a, b)| b - a == 8192));
        assert_eq!(chunk_bounds(269, 96), vec![(0, 89), (89, 179), (179, 269)]);
    }

    #[test]
    fn windows_cover_the_chunk() {
        assert_eq!(window_spans(100, 512, 128), vec![(0, 100)]);
        assert_eq!(window_spans(512, 512, 128), vec![(0, 512)]);
        assert_eq!(window_spans(700, 512, 128), vec![(0, 512), (128, 640), (188, 700)]);
        assert_eq!(window_spans(768, 512, 128), vec![(0, 512), (128, 640), (256, 768)]);
        assert_eq!(cell_spans(300, 128), vec![(0, 128), (128, 256), (256, 300)]);
    }

    #[test]
    fn pooled_spans_are_differences() {
        let (d, length) = (3, 40);
        let mut pool = ChunkPool::new(d, length, 16, 8);
        let rows: Vec<[f32; 3]> = (0..length).map(|i| [i as f32 * 0.5, (i % 7) as f32 - 3.0, 1.0]).collect();
        for row in &rows {
            pool.add(row);
        }
        let mut out = vec![0f32; 2 * d];
        pool.pooled(8, 24, &mut out);
        let mean: f64 = (8..24).map(|i| rows[i][0] as f64).sum::<f64>() / 16.0;
        assert!((out[0] as f64 - mean).abs() < 1e-6);
        assert_eq!(out[2], 1.0);
        assert_eq!(out[5], 0.0);
    }
}

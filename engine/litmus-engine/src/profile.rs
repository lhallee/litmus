//! Time spent in each stage of the encoder, summed over the process. Compiled only with the `profile` feature, which
//! no release build sets: `timed!` then wraps a stage in two clock reads, and without the feature is the stage alone.

use std::sync::atomic::{AtomicU64, Ordering};

/// The encoder's stages. Everything a scoring call does outside them (the head, the bank, the readings, copying rows
/// between layers, tokenizing) is the call's time less their sum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Stage {
    /// Reading token rows from the int8 embedding table.
    Embed,
    /// RMS norm before every ternary product, and the final norm.
    Norm,
    /// Activations to int8 with one scale per token.
    Quantize,
    /// The ternary products: adds and subtracts.
    Ternary,
    /// Rotary positions on queries and keys.
    Rotary,
    /// Attention inside a block: scores, softmax, the weighted sum of values.
    Attention,
    /// The feed-forward's squared ReLU.
    Activation,
    /// The running sums a chunk and its windows are pooled from.
    Pool,
}

pub const STAGES: [(Stage, &str); 8] = [
    (Stage::Embed, "embed"),
    (Stage::Norm, "norm"),
    (Stage::Quantize, "quantize"),
    (Stage::Ternary, "ternary"),
    (Stage::Rotary, "rotary"),
    (Stage::Attention, "attention"),
    (Stage::Activation, "activation"),
    (Stage::Pool, "pool"),
];

static NANOS: [AtomicU64; STAGES.len()] = [const { AtomicU64::new(0) }; STAGES.len()];

pub fn add(stage: Stage, nanos: u64) {
    NANOS[stage as usize].fetch_add(nanos, Ordering::Relaxed);
}

/// Each stage's nanoseconds since the last call, in `STAGES` order; the sums restart at zero.
pub fn take() -> [u64; STAGES.len()] {
    std::array::from_fn(|index| NANOS[index].swap(0, Ordering::Relaxed))
}

/// Layers counted by [`count_down`]; deeper layers add to the last.
pub const COUNTED_LAYERS: usize = 32;

/// Per layer: the `down` product's int8 inputs that were zero, all of them, the tokens, and the tokens that took the
/// column path.
static DOWN: [[AtomicU64; 4]; COUNTED_LAYERS] = [const { [const { AtomicU64::new(0) }; 4] }; COUNTED_LAYERS];

/// One `down` product of layer `layer`: `zeros` of `values` inputs were zero, over `tokens` tokens, `column_tokens`
/// of which took the column path.
pub fn count_down(layer: usize, zeros: u64, values: u64, tokens: u64, column_tokens: u64) {
    let counts = &DOWN[layer.min(COUNTED_LAYERS - 1)];
    for (counter, value) in counts.iter().zip([zeros, values, tokens, column_tokens]) {
        counter.fetch_add(value, Ordering::Relaxed);
    }
}

/// `[zeros, values, tokens, column tokens]` per layer since the last call, for the first `depth` layers; the counts
/// restart at zero.
pub fn take_down(depth: usize) -> Vec<[u64; 4]> {
    DOWN.iter().take(depth.min(COUNTED_LAYERS)).map(|counts| std::array::from_fn(|i| counts[i].swap(0, Ordering::Relaxed))).collect()
}

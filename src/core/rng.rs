//! Deterministic random number generation with forking for MCTS.
//!
//! ## Key Features
//!
//! - **Deterministic**: Same seed produces identical sequence
//! - **Forkable**: Create independent branches for MCTS simulations
//! - **Serializable**: O(1) state capture and restore
//! - **Context streams**: Independent sequences for different purposes
//!
//! ## MCTS Usage
//!
//! ```
//! use rust_ccg::core::GameRng;
//!
//! let mut rng = GameRng::new(42);
//!
//! // Fork for simulation branch
//! let mut sim_rng = rng.fork();
//!
//! // Original and fork produce different sequences
//! assert_ne!(rng.gen_range(0..100), sim_rng.gen_range(0..100));
//!
//! // But forks are deterministic - same fork counter = same sequence
//! let mut rng2 = GameRng::new(42);
//! let mut sim_rng2 = rng2.fork();
//! // sim_rng and sim_rng2 would produce the same sequence
//! ```

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

/// Deterministic RNG with forking for MCTS simulations.
///
/// Uses ChaCha8 for speed while maintaining cryptographic quality randomness.
/// Supports forking for MCTS branches and context-based independent streams.
#[derive(Clone, Debug)]
pub struct GameRng {
    inner: ChaCha8Rng,
    seed: u64,
    fork_counter: u64,
}

impl GameRng {
    /// Create a new RNG with the given seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            inner: ChaCha8Rng::seed_from_u64(seed),
            seed,
            fork_counter: 0,
        }
    }

    /// Fork this RNG to create an independent branch.
    ///
    /// Each fork produces a different but deterministic sequence.
    /// Used for MCTS simulation branches.
    #[must_use]
    pub fn fork(&mut self) -> Self {
        self.fork_counter += 1;
        let fork_seed = self.seed.wrapping_add(self.fork_counter.wrapping_mul(0x9E3779B97F4A7C15));
        Self {
            inner: ChaCha8Rng::seed_from_u64(fork_seed),
            seed: fork_seed,
            fork_counter: 0,
        }
    }

    /// Create an independent stream for a specific context.
    ///
    /// Useful for separating randomness domains (e.g., deck shuffling vs damage).
    /// The same context always produces the same stream from the same RNG state.
    #[must_use]
    pub fn for_context(&self, context: &str) -> Self {
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        self.seed.hash(&mut hasher);
        context.hash(&mut hasher);
        let context_seed = hasher.finish();

        Self {
            inner: ChaCha8Rng::seed_from_u64(context_seed),
            seed: context_seed,
            fork_counter: 0,
        }
    }

    /// Generate a random integer in the given range.
    pub fn gen_range(&mut self, range: std::ops::Range<i32>) -> i32 {
        self.inner.gen_range(range)
    }

    /// Generate a random usize in the given range.
    pub fn gen_range_usize(&mut self, range: std::ops::Range<usize>) -> usize {
        self.inner.gen_range(range)
    }

    /// Generate a random boolean with given probability of true.
    pub fn gen_bool(&mut self, probability: f64) -> bool {
        self.inner.gen_bool(probability)
    }

    /// Shuffle a slice in place.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        use rand::seq::SliceRandom;
        slice.shuffle(&mut self.inner);
    }

    /// Choose a random element from a slice.
    #[must_use]
    pub fn choose<'a, T>(&mut self, slice: &'a [T]) -> Option<&'a T> {
        use rand::seq::SliceRandom;
        slice.choose(&mut self.inner)
    }

    /// Choose a random element with weighted probability.
    ///
    /// Returns the index of the chosen element.
    /// Weights do not need to sum to 1.0.
    ///
    /// Returns `None` if weights are empty or all zero.
    pub fn choose_weighted(&mut self, weights: &[f32]) -> Option<usize> {
        if weights.is_empty() {
            return None;
        }

        let total: f32 = weights.iter().sum();
        if total <= 0.0 {
            return None;
        }

        let mut threshold = self.inner.gen::<f32>() * total;

        for (i, &weight) in weights.iter().enumerate() {
            threshold -= weight;
            if threshold <= 0.0 {
                return Some(i);
            }
        }

        // Floating point edge case - return last non-zero weight
        Some(weights.len() - 1)
    }

    /// Get the current state for serialization.
    #[must_use]
    pub fn state(&self) -> GameRngState {
        GameRngState {
            seed: self.seed,
            word_pos: self.inner.get_word_pos(),
            fork_counter: self.fork_counter,
        }
    }

    /// Restore from a saved state.
    #[must_use]
    pub fn from_state(state: &GameRngState) -> Self {
        let mut inner = ChaCha8Rng::seed_from_u64(state.seed);
        inner.set_word_pos(state.word_pos);
        Self {
            inner,
            seed: state.seed,
            fork_counter: state.fork_counter,
        }
    }
}

/// Serializable RNG state for checkpointing.
///
/// Uses ChaCha8 word position for O(1) serialization regardless of
/// how many random numbers have been generated.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GameRngState {
    /// Original seed
    pub seed: u64,
    /// ChaCha8 word position (128-bit counter)
    pub word_pos: u128,
    /// Fork counter for deterministic branching
    pub fork_counter: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determinism() {
        let mut rng1 = GameRng::new(42);
        let mut rng2 = GameRng::new(42);

        for _ in 0..100 {
            assert_eq!(rng1.gen_range(0..1000), rng2.gen_range(0..1000));
        }
    }

    #[test]
    fn test_different_seeds() {
        let mut rng1 = GameRng::new(1);
        let mut rng2 = GameRng::new(2);

        let seq1: Vec<_> = (0..10).map(|_| rng1.gen_range(0..1000)).collect();
        let seq2: Vec<_> = (0..10).map(|_| rng2.gen_range(0..1000)).collect();

        assert_ne!(seq1, seq2);
    }

    #[test]
    fn test_fork_produces_different_sequence() {
        let mut rng = GameRng::new(42);
        let mut forked = rng.fork();

        let seq1: Vec<_> = (0..10).map(|_| rng.gen_range(0..1000)).collect();
        let seq2: Vec<_> = (0..10).map(|_| forked.gen_range(0..1000)).collect();

        assert_ne!(seq1, seq2);
    }

    #[test]
    fn test_fork_is_deterministic() {
        let mut rng1 = GameRng::new(42);
        let mut rng2 = GameRng::new(42);

        let forked1 = rng1.fork();
        let forked2 = rng2.fork();

        assert_eq!(forked1.seed, forked2.seed);
    }

    #[test]
    fn test_context_produces_different_sequence() {
        let rng = GameRng::new(42);
        let mut ctx1 = rng.for_context("shuffle");
        let mut ctx2 = rng.for_context("damage");

        let seq1: Vec<_> = (0..10).map(|_| ctx1.gen_range(0..1000)).collect();
        let seq2: Vec<_> = (0..10).map(|_| ctx2.gen_range(0..1000)).collect();

        assert_ne!(seq1, seq2);
    }

    #[test]
    fn test_context_is_deterministic() {
        let rng1 = GameRng::new(42);
        let rng2 = GameRng::new(42);

        let mut ctx1 = rng1.for_context("test");
        let mut ctx2 = rng2.for_context("test");

        for _ in 0..10 {
            assert_eq!(ctx1.gen_range(0..1000), ctx2.gen_range(0..1000));
        }
    }

    #[test]
    fn test_shuffle() {
        let mut rng = GameRng::new(42);
        let mut data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let original = data.clone();

        rng.shuffle(&mut data);

        // Should be same elements, different order (very likely)
        assert_eq!(data.len(), original.len());
        assert_ne!(data, original);

        data.sort();
        assert_eq!(data, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn test_choose() {
        let mut rng = GameRng::new(42);
        let items = vec![1, 2, 3, 4, 5];

        let chosen = rng.choose(&items);
        assert!(chosen.is_some());
        assert!(items.contains(chosen.unwrap()));

        let empty: Vec<i32> = vec![];
        assert!(rng.choose(&empty).is_none());
    }

    #[test]
    fn test_choose_weighted() {
        let mut rng = GameRng::new(42);

        // Heavily weighted towards index 0
        let weights = vec![100.0, 0.0, 0.0];
        for _ in 0..10 {
            assert_eq!(rng.choose_weighted(&weights), Some(0));
        }

        // Empty weights
        assert_eq!(rng.choose_weighted(&[]), None);

        // All zero weights
        assert_eq!(rng.choose_weighted(&[0.0, 0.0]), None);
    }

    #[test]
    fn test_state_serialization() {
        let mut rng = GameRng::new(42);

        // Advance the RNG
        for _ in 0..100 {
            rng.gen_range(0..1000);
        }

        // Save state
        let state = rng.state();

        // Continue generating
        let expected: Vec<_> = (0..10).map(|_| rng.gen_range(0..1000)).collect();

        // Restore and verify
        let mut restored = GameRng::from_state(&state);
        let actual: Vec<_> = (0..10).map(|_| restored.gen_range(0..1000)).collect();

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_state_serde() {
        let state = GameRngState {
            seed: 42,
            word_pos: 12345,
            fork_counter: 5,
        };

        let json = serde_json::to_string(&state).unwrap();
        let deserialized: GameRngState = serde_json::from_str(&json).unwrap();

        assert_eq!(state, deserialized);
    }

    #[test]
    fn test_state_preserves_fork_counter() {
        let mut rng = GameRng::new(42);

        // Fork a few times
        let _ = rng.fork();
        let _ = rng.fork();
        let _ = rng.fork();

        let state = rng.state();
        assert_eq!(state.fork_counter, 3);

        let restored = GameRng::from_state(&state);
        assert_eq!(restored.fork_counter, 3);
    }
}

//! MCTS search statistics for diagnostics and tuning.

use serde::{Deserialize, Serialize};

/// Statistics collected during MCTS search.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SearchStats {
    /// Total iterations performed.
    pub iterations: u32,

    /// Nodes expanded (added to tree).
    pub nodes_expanded: u32,

    /// Simulations (rollouts) performed.
    pub simulations: u32,

    /// Maximum depth reached during search.
    pub max_depth: u16,

    /// Total time spent searching (microseconds).
    pub time_us: u64,
}

impl SearchStats {
    /// Create new empty statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset all statistics to zero.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Calculate iterations per second.
    #[must_use]
    pub fn iterations_per_second(&self) -> f64 {
        if self.time_us == 0 {
            0.0
        } else {
            self.iterations as f64 / (self.time_us as f64 / 1_000_000.0)
        }
    }

    /// Calculate simulations per second.
    #[must_use]
    pub fn simulations_per_second(&self) -> f64 {
        if self.time_us == 0 {
            0.0
        } else {
            self.simulations as f64 / (self.time_us as f64 / 1_000_000.0)
        }
    }

    /// Calculate average depth reached.
    #[must_use]
    pub fn avg_nodes_per_iteration(&self) -> f64 {
        if self.iterations == 0 {
            0.0
        } else {
            self.nodes_expanded as f64 / self.iterations as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_default() {
        let stats = SearchStats::new();
        assert_eq!(stats.iterations, 0);
        assert_eq!(stats.simulations, 0);
    }

    #[test]
    fn test_stats_iterations_per_second() {
        let mut stats = SearchStats::new();
        stats.iterations = 1000;
        stats.time_us = 1_000_000; // 1 second

        assert_eq!(stats.iterations_per_second(), 1000.0);
    }

    #[test]
    fn test_stats_reset() {
        let mut stats = SearchStats::new();
        stats.iterations = 100;
        stats.simulations = 50;

        stats.reset();

        assert_eq!(stats.iterations, 0);
        assert_eq!(stats.simulations, 0);
    }

    #[test]
    fn test_stats_serialization() {
        let mut stats = SearchStats::new();
        stats.iterations = 42;

        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: SearchStats = serde_json::from_str(&json).unwrap();

        assert_eq!(stats.iterations, deserialized.iterations);
    }
}

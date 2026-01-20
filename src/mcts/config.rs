//! MCTS configuration parameters.

use serde::{Deserialize, Serialize};

/// MCTS configuration parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MCTSConfig {
    /// UCB1 exploration constant (default: sqrt(2) = 1.414).
    /// Higher values favor exploration over exploitation.
    pub exploration_constant: f64,

    /// Maximum tree depth (0 = unlimited).
    /// Limits how deep the tree can grow.
    pub max_depth: u32,

    /// Maximum nodes to allocate in the tree.
    /// Prevents memory exhaustion on large searches.
    pub max_nodes: usize,

    /// Minimum visits before a node is expanded.
    /// Higher values delay expansion until more confident.
    pub expansion_threshold: u32,

    /// Random seed for simulation RNG.
    /// Same seed produces deterministic searches.
    pub seed: u64,

    /// Discount factor for future rewards (1.0 = no discount).
    /// Values < 1.0 prefer immediate rewards.
    pub gamma: f64,

    /// Temperature for action selection (0 = greedy, higher = more exploration).
    /// Affects final action selection from root.
    pub temperature: f64,
}

impl Default for MCTSConfig {
    fn default() -> Self {
        Self {
            exploration_constant: std::f64::consts::SQRT_2,
            max_depth: 0,
            max_nodes: 100_000,
            expansion_threshold: 1,
            seed: 42,
            gamma: 1.0,
            temperature: 0.0, // Greedy by default
        }
    }
}

impl MCTSConfig {
    /// Create a new config with custom exploration constant.
    pub fn with_exploration(mut self, c: f64) -> Self {
        self.exploration_constant = c;
        self
    }

    /// Create a new config with custom seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Create a new config with custom max depth.
    pub fn with_max_depth(mut self, depth: u32) -> Self {
        self.max_depth = depth;
        self
    }

    /// Create a new config with custom temperature.
    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.temperature = temp;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MCTSConfig::default();
        assert!((config.exploration_constant - std::f64::consts::SQRT_2).abs() < 0.001);
        assert_eq!(config.max_depth, 0);
        assert_eq!(config.seed, 42);
        assert_eq!(config.temperature, 0.0);
    }

    #[test]
    fn test_builder_pattern() {
        let config = MCTSConfig::default()
            .with_exploration(2.0)
            .with_seed(123)
            .with_max_depth(50);

        assert_eq!(config.exploration_constant, 2.0);
        assert_eq!(config.seed, 123);
        assert_eq!(config.max_depth, 50);
    }

    #[test]
    fn test_serialization() {
        let config = MCTSConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: MCTSConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.seed, deserialized.seed);
    }
}

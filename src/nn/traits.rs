//! Neural network traits for policy and value prediction.
//!
//! These traits define the interface between the Rust game engine and
//! neural network implementations (typically in Python via PyO3).

use serde::{Deserialize, Serialize};

/// Encoded game state as a flat tensor for neural network input.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncodedState {
    /// Flattened tensor data (row-major order).
    pub tensor: Vec<f32>,

    /// Shape of the tensor (e.g., [channels, height, width] or [features]).
    pub shape: Vec<usize>,
}

impl EncodedState {
    /// Create a new encoded state.
    pub fn new(tensor: Vec<f32>, shape: Vec<usize>) -> Self {
        debug_assert_eq!(
            tensor.len(),
            shape.iter().product::<usize>(),
            "Tensor length must match shape product"
        );
        Self { tensor, shape }
    }

    /// Create a zero-filled encoded state with the given shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self {
            tensor: vec![0.0; size],
            shape,
        }
    }

    /// Get the total number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensor.len()
    }

    /// Check if the tensor is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensor.is_empty()
    }

    /// Get element at a flat index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<f32> {
        self.tensor.get(index).copied()
    }

    /// Set element at a flat index.
    pub fn set(&mut self, index: usize, value: f32) {
        if index < self.tensor.len() {
            self.tensor[index] = value;
        }
    }
}

/// Policy network outputs action probabilities.
///
/// Given an encoded state, returns a probability distribution over actions.
/// The output vector length should match the action space size.
pub trait PolicyNetwork: Send + Sync {
    /// Predict action probabilities for the given state.
    ///
    /// Returns a vector of probabilities (should sum to 1.0).
    /// Length should equal `action_space_size` from the encoder.
    fn predict(&self, encoded: &EncodedState) -> Vec<f32>;

    /// Batch prediction for multiple states (optional optimization).
    fn predict_batch(&self, encoded: &[EncodedState]) -> Vec<Vec<f32>> {
        encoded.iter().map(|e| self.predict(e)).collect()
    }
}

/// Value network outputs per-player value estimates.
///
/// Given an encoded state, returns expected rewards for each player.
pub trait ValueNetwork: Send + Sync {
    /// Predict value for the given state.
    ///
    /// Returns a vector of values, one per player.
    /// Values are typically in [-1, 1] or [0, 1] range.
    fn predict(&self, encoded: &EncodedState) -> Vec<f32>;

    /// Batch prediction for multiple states (optional optimization).
    fn predict_batch(&self, encoded: &[EncodedState]) -> Vec<Vec<f32>> {
        encoded.iter().map(|e| self.predict(e)).collect()
    }
}

/// Combined policy-value network (more efficient than separate networks).
///
/// Many architectures share early layers between policy and value heads,
/// so this trait allows a single forward pass for both outputs.
pub trait PolicyValueNetwork: Send + Sync {
    /// Predict both policy and value for the given state.
    ///
    /// Returns (policy_probs, player_values).
    fn predict(&self, encoded: &EncodedState) -> (Vec<f32>, Vec<f32>);

    /// Batch prediction for multiple states (optional optimization).
    fn predict_batch(&self, encoded: &[EncodedState]) -> Vec<(Vec<f32>, Vec<f32>)> {
        encoded.iter().map(|e| self.predict(e)).collect()
    }
}

/// Uniform random policy (baseline for testing).
#[derive(Clone, Debug, Default)]
pub struct UniformPolicy {
    action_space_size: usize,
}

impl UniformPolicy {
    /// Create a new uniform policy.
    pub fn new(action_space_size: usize) -> Self {
        Self { action_space_size }
    }
}

impl PolicyNetwork for UniformPolicy {
    fn predict(&self, _encoded: &EncodedState) -> Vec<f32> {
        if self.action_space_size == 0 {
            return vec![];
        }
        let prob = 1.0 / self.action_space_size as f32;
        vec![prob; self.action_space_size]
    }
}

/// Zero value network (baseline for testing).
#[derive(Clone, Debug, Default)]
pub struct ZeroValue {
    player_count: usize,
}

impl ZeroValue {
    /// Create a new zero value network.
    pub fn new(player_count: usize) -> Self {
        Self { player_count }
    }
}

impl ValueNetwork for ZeroValue {
    fn predict(&self, _encoded: &EncodedState) -> Vec<f32> {
        vec![0.0; self.player_count]
    }
}

/// Combined uniform policy and zero value network.
#[derive(Clone, Debug, Default)]
pub struct UniformPolicyZeroValue {
    action_space_size: usize,
    player_count: usize,
}

impl UniformPolicyZeroValue {
    /// Create a new baseline network.
    pub fn new(action_space_size: usize, player_count: usize) -> Self {
        Self {
            action_space_size,
            player_count,
        }
    }
}

impl PolicyValueNetwork for UniformPolicyZeroValue {
    fn predict(&self, _encoded: &EncodedState) -> (Vec<f32>, Vec<f32>) {
        let policy = if self.action_space_size == 0 {
            vec![]
        } else {
            let prob = 1.0 / self.action_space_size as f32;
            vec![prob; self.action_space_size]
        };
        let value = vec![0.0; self.player_count];
        (policy, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoded_state_new() {
        let state = EncodedState::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(state.len(), 4);
        assert_eq!(state.shape, vec![2, 2]);
        assert_eq!(state.get(0), Some(1.0));
        assert_eq!(state.get(3), Some(4.0));
        assert_eq!(state.get(4), None);
    }

    #[test]
    fn test_encoded_state_zeros() {
        let state = EncodedState::zeros(vec![3, 4]);
        assert_eq!(state.len(), 12);
        assert!(state.tensor.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_encoded_state_set() {
        let mut state = EncodedState::zeros(vec![4]);
        state.set(2, 5.0);
        assert_eq!(state.tensor, vec![0.0, 0.0, 5.0, 0.0]);
    }

    #[test]
    fn test_uniform_policy() {
        let policy = UniformPolicy::new(4);
        let state = EncodedState::zeros(vec![10]);
        let probs = policy.predict(&state);

        assert_eq!(probs.len(), 4);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 0.001);
        assert!(probs.iter().all(|&p| (p - 0.25).abs() < 0.001));
    }

    #[test]
    fn test_zero_value() {
        let value = ZeroValue::new(3);
        let state = EncodedState::zeros(vec![10]);
        let values = value.predict(&state);

        assert_eq!(values.len(), 3);
        assert!(values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_uniform_policy_zero_value() {
        let network = UniformPolicyZeroValue::new(5, 2);
        let state = EncodedState::zeros(vec![10]);
        let (policy, value) = network.predict(&state);

        assert_eq!(policy.len(), 5);
        assert_eq!(value.len(), 2);
        assert!((policy.iter().sum::<f32>() - 1.0).abs() < 0.001);
        assert!(value.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_serialization() {
        let state = EncodedState::new(vec![1.0, 2.0, 3.0], vec![3]);
        let json = serde_json::to_string(&state).unwrap();
        let deserialized: EncodedState = serde_json::from_str(&json).unwrap();

        assert_eq!(state.tensor, deserialized.tensor);
        assert_eq!(state.shape, deserialized.shape);
    }

    #[test]
    fn test_encoded_state_is_empty() {
        let empty = EncodedState::zeros(vec![0]);
        assert!(empty.is_empty());

        let non_empty = EncodedState::zeros(vec![3]);
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_encoded_state_set_out_of_bounds() {
        let mut state = EncodedState::zeros(vec![3]);
        state.set(10, 5.0); // Out of bounds, should be ignored
        assert!(state.tensor.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_uniform_policy_zero_actions() {
        let policy = UniformPolicy::new(0);
        let state = EncodedState::zeros(vec![10]);
        let probs = policy.predict(&state);

        assert!(probs.is_empty());
    }

    #[test]
    fn test_uniform_policy_single_action() {
        let policy = UniformPolicy::new(1);
        let state = EncodedState::zeros(vec![10]);
        let probs = policy.predict(&state);

        assert_eq!(probs.len(), 1);
        assert_eq!(probs[0], 1.0);
    }

    #[test]
    fn test_policy_network_batch() {
        let policy = UniformPolicy::new(3);
        let states = vec![
            EncodedState::zeros(vec![10]),
            EncodedState::zeros(vec![10]),
            EncodedState::zeros(vec![10]),
        ];

        let batch_results = policy.predict_batch(&states);
        assert_eq!(batch_results.len(), 3);

        for probs in &batch_results {
            assert_eq!(probs.len(), 3);
            assert!((probs.iter().sum::<f32>() - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_value_network_batch() {
        let value = ZeroValue::new(4);
        let states = vec![
            EncodedState::zeros(vec![10]),
            EncodedState::zeros(vec![10]),
        ];

        let batch_results = value.predict_batch(&states);
        assert_eq!(batch_results.len(), 2);

        for values in &batch_results {
            assert_eq!(values.len(), 4);
            assert!(values.iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn test_policy_value_network_batch() {
        let network = UniformPolicyZeroValue::new(5, 2);
        let states = vec![
            EncodedState::zeros(vec![10]),
            EncodedState::zeros(vec![10]),
            EncodedState::zeros(vec![10]),
        ];

        let batch_results = network.predict_batch(&states);
        assert_eq!(batch_results.len(), 3);

        for (policy, value) in &batch_results {
            assert_eq!(policy.len(), 5);
            assert_eq!(value.len(), 2);
        }
    }

    #[test]
    fn test_uniform_policy_zero_value_zero_actions() {
        let network = UniformPolicyZeroValue::new(0, 2);
        let state = EncodedState::zeros(vec![10]);
        let (policy, value) = network.predict(&state);

        assert!(policy.is_empty());
        assert_eq!(value.len(), 2);
    }

    #[test]
    fn test_zero_value_single_player() {
        let value = ZeroValue::new(1);
        let state = EncodedState::zeros(vec![5]);
        let values = value.predict(&state);

        assert_eq!(values.len(), 1);
        assert_eq!(values[0], 0.0);
    }

    #[test]
    fn test_uniform_policy_default() {
        let policy = UniformPolicy::default();
        let state = EncodedState::zeros(vec![5]);
        let probs = policy.predict(&state);

        // Default has action_space_size = 0
        assert!(probs.is_empty());
    }

    #[test]
    fn test_zero_value_default() {
        let value = ZeroValue::default();
        let state = EncodedState::zeros(vec![5]);
        let values = value.predict(&state);

        // Default has player_count = 0
        assert!(values.is_empty());
    }

    #[test]
    fn test_uniform_policy_zero_value_default() {
        let network = UniformPolicyZeroValue::default();
        let state = EncodedState::zeros(vec![5]);
        let (policy, value) = network.predict(&state);

        assert!(policy.is_empty());
        assert!(value.is_empty());
    }

    #[test]
    fn test_encoded_state_multidimensional_shape() {
        let state = EncodedState::new(vec![0.0; 24], vec![2, 3, 4]);
        assert_eq!(state.len(), 24);
        assert_eq!(state.shape, vec![2, 3, 4]);
    }
}

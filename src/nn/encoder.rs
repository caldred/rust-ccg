//! State encoding for neural network input.
//!
//! Transforms game state into tensor representations suitable for neural networks.

use crate::core::{GameState, PlayerId};
use crate::nn::traits::EncodedState;

/// Encodes game state into tensors for neural network input.
///
/// Each encoder defines:
/// - How to convert state to a tensor from a player's perspective
/// - The shape of the output tensor
/// - The size of the action space
pub trait StateEncoder: Send + Sync {
    /// Encode the game state from a player's perspective.
    ///
    /// The encoding should hide information not visible to the player
    /// (e.g., opponent's hand contents, face-down cards).
    fn encode(&self, state: &GameState, perspective: PlayerId) -> EncodedState;

    /// Get the shape of encoded states.
    fn output_shape(&self) -> Vec<usize>;

    /// Get the total number of possible actions.
    ///
    /// This defines the size of the policy output vector.
    fn action_space_size(&self) -> usize;

    /// Get the number of players this encoder supports.
    fn player_count(&self) -> usize;
}

/// Simple encoder for the SimpleGame.
///
/// Encodes state as a flat vector:
/// - Player lives (normalized to [0, 1])
/// - Hand sizes
/// - Deck sizes
/// - Current player indicator
/// - Perspective player indicator
///
/// Total features = 5 * player_count
#[derive(Clone, Debug)]
pub struct SimpleGameEncoder {
    player_count: usize,
    max_life: f32,
    max_cards: f32,
    action_space: usize,
}

impl SimpleGameEncoder {
    /// Create a new SimpleGame encoder.
    ///
    /// # Arguments
    /// - `player_count`: Number of players in the game
    /// - `max_life`: Maximum life value for normalization (default: 50)
    /// - `action_space`: Size of the action space
    pub fn new(player_count: usize, action_space: usize) -> Self {
        Self {
            player_count,
            max_life: 50.0,
            max_cards: 20.0,
            action_space,
        }
    }

    /// Set the maximum life for normalization.
    pub fn with_max_life(mut self, max_life: f32) -> Self {
        self.max_life = max_life;
        self
    }

    /// Set the maximum cards for normalization.
    pub fn with_max_cards(mut self, max_cards: f32) -> Self {
        self.max_cards = max_cards;
        self
    }

    /// Get the number of features per player.
    fn features_per_player(&self) -> usize {
        5 // life, hand_size, deck_size, is_active, is_perspective
    }
}

impl StateEncoder for SimpleGameEncoder {
    fn encode(&self, state: &GameState, perspective: PlayerId) -> EncodedState {
        let features_per_player = self.features_per_player();
        let total_features = features_per_player * self.player_count;
        let mut tensor = vec![0.0f32; total_features];

        for player_idx in 0..self.player_count {
            let player = PlayerId::new(player_idx as u8);
            let base_idx = player_idx * features_per_player;

            // Life (normalized)
            let life = state.public.get_player_state(player, "life", 0) as f32;
            tensor[base_idx] = (life / self.max_life).clamp(0.0, 1.0);

            // Hand size (normalized)
            let hand_size = state.public.hand_sizes[player] as f32;
            tensor[base_idx + 1] = (hand_size / self.max_cards).clamp(0.0, 1.0);

            // Deck size - we use a placeholder since we don't have direct access
            // In a real implementation, this would query the zone sizes
            tensor[base_idx + 2] = 0.5; // Placeholder

            // Is active player
            tensor[base_idx + 3] = if state.public.active_player == player {
                1.0
            } else {
                0.0
            };

            // Is perspective player
            tensor[base_idx + 4] = if player == perspective { 1.0 } else { 0.0 };
        }

        EncodedState::new(tensor, vec![total_features])
    }

    fn output_shape(&self) -> Vec<usize> {
        vec![self.features_per_player() * self.player_count]
    }

    fn action_space_size(&self) -> usize {
        self.action_space
    }

    fn player_count(&self) -> usize {
        self.player_count
    }
}

/// Encoder that produces a fixed-size zero tensor (for testing).
#[derive(Clone, Debug)]
pub struct ZeroEncoder {
    shape: Vec<usize>,
    action_space: usize,
    player_count: usize,
}

impl ZeroEncoder {
    /// Create a new zero encoder.
    pub fn new(shape: Vec<usize>, action_space: usize, player_count: usize) -> Self {
        Self {
            shape,
            action_space,
            player_count,
        }
    }
}

impl StateEncoder for ZeroEncoder {
    fn encode(&self, _state: &GameState, _perspective: PlayerId) -> EncodedState {
        EncodedState::zeros(self.shape.clone())
    }

    fn output_shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn action_space_size(&self) -> usize {
        self.action_space
    }

    fn player_count(&self) -> usize {
        self.player_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::simple::SimpleGameBuilder;

    #[test]
    fn test_simple_game_encoder_shape() {
        let encoder = SimpleGameEncoder::new(2, 10);
        let shape = encoder.output_shape();

        // 5 features per player * 2 players = 10
        assert_eq!(shape, vec![10]);
        assert_eq!(encoder.action_space_size(), 10);
        assert_eq!(encoder.player_count(), 2);
    }

    #[test]
    fn test_simple_game_encoder_encode() {
        let (_game, state) = SimpleGameBuilder::new()
            .player_count(2)
            .starting_life(20)
            .build(42);

        let encoder = SimpleGameEncoder::new(2, 10).with_max_life(20.0);
        let encoded = encoder.encode(&state, PlayerId::new(0));

        assert_eq!(encoded.len(), 10);
        assert_eq!(encoded.shape, vec![10]);

        // Player 0 should have life = 20/20 = 1.0
        assert!((encoded.tensor[0] - 1.0).abs() < 0.01);

        // Player 0 is perspective player (index 4)
        assert_eq!(encoded.tensor[4], 1.0);

        // Player 1 is not perspective player (index 9)
        assert_eq!(encoded.tensor[9], 0.0);
    }

    #[test]
    fn test_simple_game_encoder_active_player() {
        let (_game, state) = SimpleGameBuilder::new()
            .player_count(2)
            .build(42);

        let encoder = SimpleGameEncoder::new(2, 10);
        let encoded = encoder.encode(&state, PlayerId::new(1));

        // Player 0 is active by default
        assert_eq!(encoded.tensor[3], 1.0); // Player 0 is_active
        assert_eq!(encoded.tensor[8], 0.0); // Player 1 is_active

        // Perspective is player 1
        assert_eq!(encoded.tensor[4], 0.0); // Player 0 is_perspective
        assert_eq!(encoded.tensor[9], 1.0); // Player 1 is_perspective
    }

    #[test]
    fn test_four_player_encoder() {
        let (_game, state) = SimpleGameBuilder::new()
            .player_count(4)
            .starting_life(15)
            .build(42);

        let encoder = SimpleGameEncoder::new(4, 20);
        let encoded = encoder.encode(&state, PlayerId::new(2));

        // 5 features * 4 players = 20
        assert_eq!(encoded.len(), 20);

        // Perspective player indicator at index 14 (player 2, feature 4)
        assert_eq!(encoded.tensor[14], 1.0);
    }

    #[test]
    fn test_zero_encoder() {
        let (_game, state) = SimpleGameBuilder::new()
            .player_count(2)
            .build(42);

        let encoder = ZeroEncoder::new(vec![3, 4], 10, 2);
        let encoded = encoder.encode(&state, PlayerId::new(0));

        assert_eq!(encoded.len(), 12);
        assert_eq!(encoded.shape, vec![3, 4]);
        assert!(encoded.tensor.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_encoder_serialization() {
        let (_game, state) = SimpleGameBuilder::new()
            .player_count(2)
            .build(42);

        let encoder = SimpleGameEncoder::new(2, 10);
        let encoded = encoder.encode(&state, PlayerId::new(0));

        let json = serde_json::to_string(&encoded).unwrap();
        let deserialized: EncodedState = serde_json::from_str(&json).unwrap();

        assert_eq!(encoded.tensor, deserialized.tensor);
        assert_eq!(encoded.shape, deserialized.shape);
    }

    #[test]
    fn test_simple_game_encoder_with_max_cards() {
        let (_game, state) = SimpleGameBuilder::new()
            .player_count(2)
            .starting_hand_size(5)
            .build(42);

        let encoder = SimpleGameEncoder::new(2, 10).with_max_cards(10.0);
        let encoded = encoder.encode(&state, PlayerId::new(0));

        // Hand size = 5, max = 10, so normalized = 0.5
        // Hand size is at index 1 (second feature for player 0)
        assert!((encoded.tensor[1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_simple_game_encoder_clamping() {
        let (_game, state) = SimpleGameBuilder::new()
            .player_count(2)
            .starting_life(100)
            .build(42);

        // Set max_life low to test clamping
        let encoder = SimpleGameEncoder::new(2, 10).with_max_life(10.0);
        let encoded = encoder.encode(&state, PlayerId::new(0));

        // Life = 100, max = 10, should clamp to 1.0
        assert_eq!(encoded.tensor[0], 1.0);
    }

    #[test]
    fn test_simple_game_encoder_hand_size_normalization() {
        let (_game, state) = SimpleGameBuilder::new()
            .player_count(2)
            .starting_hand_size(10)
            .build(42);

        let encoder = SimpleGameEncoder::new(2, 10).with_max_cards(5.0);
        let encoded = encoder.encode(&state, PlayerId::new(0));

        // Hand size > max, should clamp to 1.0
        assert_eq!(encoded.tensor[1], 1.0);
    }

    #[test]
    fn test_simple_game_encoder_six_players() {
        let (_game, state) = SimpleGameBuilder::new()
            .player_count(6)
            .starting_life(20)
            .build(42);

        let encoder = SimpleGameEncoder::new(6, 30);
        let encoded = encoder.encode(&state, PlayerId::new(3));

        // 5 features * 6 players = 30
        assert_eq!(encoded.len(), 30);
        assert_eq!(encoder.output_shape(), vec![30]);

        // Player 3 is perspective player (feature 4 for player 3 = index 19)
        assert_eq!(encoded.tensor[19], 1.0);

        // Other players are not perspective
        assert_eq!(encoded.tensor[4], 0.0);  // Player 0
        assert_eq!(encoded.tensor[9], 0.0);  // Player 1
        assert_eq!(encoded.tensor[14], 0.0); // Player 2
        assert_eq!(encoded.tensor[24], 0.0); // Player 4
        assert_eq!(encoded.tensor[29], 0.0); // Player 5
    }

    #[test]
    fn test_simple_game_encoder_all_features() {
        let (_game, state) = SimpleGameBuilder::new()
            .player_count(2)
            .starting_life(20)
            .starting_hand_size(3)
            .build(42);

        let encoder = SimpleGameEncoder::new(2, 10)
            .with_max_life(20.0)
            .with_max_cards(10.0);
        let encoded = encoder.encode(&state, PlayerId::new(0));

        // Player 0 features:
        // [0]: life = 20/20 = 1.0
        // [1]: hand_size = 3/10 = 0.3
        // [2]: deck_size placeholder = 0.5
        // [3]: is_active = 1.0 (player 0 starts)
        // [4]: is_perspective = 1.0

        assert!((encoded.tensor[0] - 1.0).abs() < 0.01);
        assert!((encoded.tensor[1] - 0.3).abs() < 0.01);
        assert!((encoded.tensor[2] - 0.5).abs() < 0.01);
        assert_eq!(encoded.tensor[3], 1.0);
        assert_eq!(encoded.tensor[4], 1.0);

        // Player 1 features:
        // [5]: life = 20/20 = 1.0
        // [6]: hand_size = 3/10 = 0.3
        // [7]: deck_size placeholder = 0.5
        // [8]: is_active = 0.0
        // [9]: is_perspective = 0.0

        assert!((encoded.tensor[5] - 1.0).abs() < 0.01);
        assert!((encoded.tensor[6] - 0.3).abs() < 0.01);
        assert!((encoded.tensor[7] - 0.5).abs() < 0.01);
        assert_eq!(encoded.tensor[8], 0.0);
        assert_eq!(encoded.tensor[9], 0.0);
    }

    #[test]
    fn test_zero_encoder_properties() {
        let encoder = ZeroEncoder::new(vec![5, 5], 25, 4);

        assert_eq!(encoder.output_shape(), vec![5, 5]);
        assert_eq!(encoder.action_space_size(), 25);
        assert_eq!(encoder.player_count(), 4);
    }

    #[test]
    fn test_zero_encoder_different_perspectives() {
        let (_game, state) = SimpleGameBuilder::new()
            .player_count(2)
            .build(42);

        let encoder = ZeroEncoder::new(vec![10], 5, 2);

        // Different perspectives should give same result (all zeros)
        let encoded_p0 = encoder.encode(&state, PlayerId::new(0));
        let encoded_p1 = encoder.encode(&state, PlayerId::new(1));

        assert_eq!(encoded_p0.tensor, encoded_p1.tensor);
        assert!(encoded_p0.tensor.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_simple_game_encoder_damaged_player() {
        let (_game, mut state) = SimpleGameBuilder::new()
            .player_count(2)
            .starting_life(20)
            .build(42);

        // Deal 10 damage to player 1
        state.public.modify_player_state(PlayerId::new(1), "life", -10);

        let encoder = SimpleGameEncoder::new(2, 10).with_max_life(20.0);
        let encoded = encoder.encode(&state, PlayerId::new(0));

        // Player 0 life = 20/20 = 1.0
        assert!((encoded.tensor[0] - 1.0).abs() < 0.01);
        // Player 1 life = 10/20 = 0.5
        assert!((encoded.tensor[5] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_simple_game_encoder_zero_life() {
        let (_game, mut state) = SimpleGameBuilder::new()
            .player_count(2)
            .starting_life(5)
            .build(42);

        // Set player 1 to 0 life
        state.public.set_player_state(PlayerId::new(1), "life", 0);

        let encoder = SimpleGameEncoder::new(2, 10).with_max_life(20.0);
        let encoded = encoder.encode(&state, PlayerId::new(0));

        // Player 1 life = 0/20 = 0.0
        assert_eq!(encoded.tensor[5], 0.0);
    }

    #[test]
    fn test_encoder_features_per_player() {
        let encoder = SimpleGameEncoder::new(2, 10);
        // Internal method, but we can verify via output_shape
        // 5 features * 2 players = 10
        assert_eq!(encoder.output_shape(), vec![10]);

        let encoder3 = SimpleGameEncoder::new(3, 15);
        // 5 features * 3 players = 15
        assert_eq!(encoder3.output_shape(), vec![15]);
    }
}

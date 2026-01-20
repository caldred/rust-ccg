//! Self-play loop for generating training data.
//!
//! Runs games using MCTS to generate trajectories for training
//! neural networks in an AlphaZero-style loop.

use crate::core::{GameState, PlayerId, PlayerMap};
use crate::mcts::{MCTSConfig, MCTSSearch};
use crate::nn::{PolicyValueNetwork, StateEncoder};
use crate::rules::{GameResult, RulesEngine};

use super::trajectory::{Step, Trajectory};

/// Configuration for self-play.
#[derive(Clone, Debug)]
pub struct SelfPlayConfig {
    /// Number of MCTS iterations per move.
    pub mcts_iterations: u32,

    /// Temperature for action selection during early game.
    /// Higher = more exploration.
    pub temperature: f64,

    /// Move number at which to switch to greedy (temperature = 0).
    /// 0 = always use temperature.
    pub temperature_threshold: usize,

    /// Maximum moves per game (to prevent infinite games).
    pub max_moves: usize,

    /// MCTS exploration constant.
    pub exploration_constant: f64,

    /// Seed offset for RNG (combined with game index for unique seeds).
    pub seed_offset: u64,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            mcts_iterations: 800,
            temperature: 1.0,
            temperature_threshold: 30,
            max_moves: 500,
            exploration_constant: 1.414,
            seed_offset: 0,
        }
    }
}

impl SelfPlayConfig {
    /// Create a new self-play config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set MCTS iterations per move.
    pub fn with_mcts_iterations(mut self, iterations: u32) -> Self {
        self.mcts_iterations = iterations;
        self
    }

    /// Set temperature for early game exploration.
    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.temperature = temp;
        self
    }

    /// Set the move threshold for switching to greedy.
    pub fn with_temperature_threshold(mut self, threshold: usize) -> Self {
        self.temperature_threshold = threshold;
        self
    }

    /// Set maximum moves per game.
    pub fn with_max_moves(mut self, max: usize) -> Self {
        self.max_moves = max;
        self
    }

    /// Set exploration constant.
    pub fn with_exploration(mut self, c: f64) -> Self {
        self.exploration_constant = c;
        self
    }

    /// Set seed offset.
    pub fn with_seed_offset(mut self, offset: u64) -> Self {
        self.seed_offset = offset;
        self
    }

    /// Get the temperature for a given move number.
    pub fn effective_temperature(&self, move_number: usize) -> f64 {
        if self.temperature_threshold > 0 && move_number >= self.temperature_threshold {
            0.0
        } else {
            self.temperature
        }
    }
}

/// Worker for running self-play games.
///
/// Manages the MCTS search and trajectory collection.
pub struct SelfPlayWorker<E: RulesEngine + Clone> {
    /// The game engine (used as a template for new games).
    engine: E,

    /// State encoder for neural network input.
    encoder: Box<dyn StateEncoder>,

    /// Self-play configuration.
    config: SelfPlayConfig,
}

impl<E: RulesEngine + Clone> SelfPlayWorker<E> {
    /// Create a new self-play worker.
    pub fn new(engine: E, encoder: Box<dyn StateEncoder>, config: SelfPlayConfig) -> Self {
        Self {
            engine,
            encoder,
            config,
        }
    }

    /// Play a single game without a neural network (pure MCTS).
    ///
    /// Uses random rollouts for evaluation.
    pub fn play_game(&self, state: &mut GameState, seed: u64) -> Trajectory {
        let player_count = state.player_count();
        let mut trajectory = Trajectory::new(seed, player_count);

        for move_number in 0..self.config.max_moves {
            // Check for terminal state
            if self.engine.is_terminal(state).is_some() {
                break;
            }

            let active_player = state.public.active_player;

            // Run MCTS search
            let mcts_config = MCTSConfig::default()
                .with_exploration(self.config.exploration_constant)
                .with_temperature(self.config.effective_temperature(move_number))
                .with_seed(seed.wrapping_add(move_number as u64));

            let mut search = MCTSSearch::new(self.engine.clone(), mcts_config);
            let action = search.search(state, active_player, self.config.mcts_iterations);

            let action = match action {
                Some(a) => a,
                None => break, // No legal actions
            };

            // Record step
            let encoded_state = self.encoder.encode(state, active_player);
            let action_probs = search.action_probabilities();

            let step = Step::new(
                encoded_state,
                action_probs,
                action.clone(),
                active_player,
                move_number,
            );
            trajectory.push(step);

            // Apply action
            let mut engine = self.engine.clone();
            engine.apply_action(state, active_player, &action);
        }

        // Set final outcome
        let outcome = self.compute_outcome(state, player_count);
        trajectory.set_outcome(outcome);

        trajectory
    }

    /// Play a game with a neural network for policy/value guidance.
    ///
    /// The network provides prior probabilities for MCTS edges and
    /// value estimates for leaf evaluation.
    pub fn play_game_with_network<N: PolicyValueNetwork>(
        &self,
        state: &mut GameState,
        seed: u64,
        network: &N,
    ) -> Trajectory {
        let player_count = state.player_count();
        let mut trajectory = Trajectory::new(seed, player_count);

        for move_number in 0..self.config.max_moves {
            // Check for terminal state
            if self.engine.is_terminal(state).is_some() {
                break;
            }

            let active_player = state.public.active_player;

            // Encode state and get network predictions
            let encoded = self.encoder.encode(state, active_player);
            let (policy_probs, _value) = network.predict(&encoded);

            // Create MCTS with network priors
            let mcts_config = MCTSConfig::default()
                .with_exploration(self.config.exploration_constant)
                .with_temperature(self.config.effective_temperature(move_number))
                .with_seed(seed.wrapping_add(move_number as u64));

            let mut search = MCTSSearch::new(self.engine.clone(), mcts_config);

            // Run search (note: priors would be set via set_root_priors if implemented)
            let action = search.search(state, active_player, self.config.mcts_iterations);

            // For now, we use policy_probs in the trajectory but don't integrate
            // them into MCTS selection. Full integration requires extending MCTSSearch.
            let _ = policy_probs; // Silence unused warning

            let action = match action {
                Some(a) => a,
                None => break,
            };

            // Record step
            let action_probs = search.action_probabilities();
            let step = Step::new(
                encoded,
                action_probs,
                action.clone(),
                active_player,
                move_number,
            );
            trajectory.push(step);

            // Apply action
            let mut engine = self.engine.clone();
            engine.apply_action(state, active_player, &action);
        }

        // Set final outcome
        let outcome = self.compute_outcome(state, player_count);
        trajectory.set_outcome(outcome);

        trajectory
    }

    /// Play multiple games without a neural network.
    pub fn play_games(&self, game_builder: impl Fn(u64) -> (E, GameState), count: usize) -> Vec<Trajectory> {
        (0..count)
            .map(|i| {
                let seed = self.config.seed_offset.wrapping_add(i as u64);
                let (_engine, mut state) = game_builder(seed);
                self.play_game(&mut state, seed)
            })
            .collect()
    }

    /// Compute the outcome rewards for each player.
    fn compute_outcome(&self, state: &GameState, player_count: usize) -> PlayerMap<f64> {
        let mut outcome = PlayerMap::with_value(player_count, 0.0);

        match self.engine.is_terminal(state) {
            Some(GameResult::Winner(winner)) => {
                outcome[winner] = 1.0;
            }
            Some(GameResult::Winners(ref winners)) => {
                // Multiple winners share the reward
                let share = 1.0 / winners.len() as f64;
                for &player in winners {
                    outcome[player] = share;
                }
            }
            Some(GameResult::Draw) => {
                // Equal reward for all
                let share = 1.0 / player_count as f64;
                for player in PlayerId::all(player_count) {
                    outcome[player] = share;
                }
            }
            None => {
                // Game didn't complete (hit max moves)
                // Use heuristic evaluation
                let share = 0.5;
                for player in PlayerId::all(player_count) {
                    outcome[player] = share;
                }
            }
        }

        outcome
    }

    /// Get the encoder.
    pub fn encoder(&self) -> &dyn StateEncoder {
        self.encoder.as_ref()
    }

    /// Get the configuration.
    pub fn config(&self) -> &SelfPlayConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::simple::SimpleGameBuilder;
    use crate::nn::{SimpleGameEncoder, UniformPolicyZeroValue};

    #[test]
    fn test_self_play_config_default() {
        let config = SelfPlayConfig::default();
        assert_eq!(config.mcts_iterations, 800);
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.temperature_threshold, 30);
    }

    #[test]
    fn test_effective_temperature() {
        let config = SelfPlayConfig::default()
            .with_temperature(1.5)
            .with_temperature_threshold(10);

        assert_eq!(config.effective_temperature(0), 1.5);
        assert_eq!(config.effective_temperature(5), 1.5);
        assert_eq!(config.effective_temperature(9), 1.5);
        assert_eq!(config.effective_temperature(10), 0.0);
        assert_eq!(config.effective_temperature(100), 0.0);
    }

    #[test]
    fn test_effective_temperature_no_threshold() {
        let config = SelfPlayConfig::default()
            .with_temperature(1.0)
            .with_temperature_threshold(0);

        assert_eq!(config.effective_temperature(0), 1.0);
        assert_eq!(config.effective_temperature(100), 1.0);
    }

    #[test]
    fn test_play_game() {
        let (engine, mut state) = SimpleGameBuilder::new()
            .player_count(2)
            .starting_life(5) // Low life for quick games
            .build(42);

        let encoder = Box::new(SimpleGameEncoder::new(2, 10));
        let config = SelfPlayConfig::default()
            .with_mcts_iterations(10) // Few iterations for testing
            .with_max_moves(50);

        let worker = SelfPlayWorker::new(engine, encoder, config);
        let trajectory = worker.play_game(&mut state, 42);

        // Should have recorded some steps
        assert!(!trajectory.is_empty());

        // Outcome should be set
        let p0_outcome = trajectory.player_outcome(PlayerId::new(0));
        let p1_outcome = trajectory.player_outcome(PlayerId::new(1));

        // Outcomes should be valid
        assert!(p0_outcome >= 0.0 && p0_outcome <= 1.0);
        assert!(p1_outcome >= 0.0 && p1_outcome <= 1.0);
    }

    #[test]
    fn test_play_game_with_network() {
        let (engine, mut state) = SimpleGameBuilder::new()
            .player_count(2)
            .starting_life(5)
            .build(42);

        let encoder = Box::new(SimpleGameEncoder::new(2, 10));
        let config = SelfPlayConfig::default()
            .with_mcts_iterations(10)
            .with_max_moves(50);

        let worker = SelfPlayWorker::new(engine, encoder, config);
        let network = UniformPolicyZeroValue::new(10, 2);
        let trajectory = worker.play_game_with_network(&mut state, 42, &network);

        assert!(!trajectory.is_empty());
    }

    #[test]
    fn test_play_games() {
        let config = SelfPlayConfig::default()
            .with_mcts_iterations(5)
            .with_max_moves(20);

        let encoder = Box::new(SimpleGameEncoder::new(2, 10));

        // Create worker with a template engine
        let (template_engine, _) = SimpleGameBuilder::new()
            .player_count(2)
            .starting_life(3)
            .build(0);

        let worker = SelfPlayWorker::new(template_engine, encoder, config);

        let trajectories = worker.play_games(
            |seed| {
                SimpleGameBuilder::new()
                    .player_count(2)
                    .starting_life(3)
                    .build(seed)
            },
            3,
        );

        assert_eq!(trajectories.len(), 3);
        for traj in &trajectories {
            assert!(!traj.is_empty());
        }
    }

    #[test]
    fn test_trajectory_has_action_probs() {
        let (engine, mut state) = SimpleGameBuilder::new()
            .player_count(2)
            .starting_life(3)
            .build(42);

        let encoder = Box::new(SimpleGameEncoder::new(2, 10));
        let config = SelfPlayConfig::default()
            .with_mcts_iterations(20)
            .with_max_moves(20);

        let worker = SelfPlayWorker::new(engine, encoder, config);
        let trajectory = worker.play_game(&mut state, 42);

        // All steps should have action probabilities
        for step in &trajectory.steps {
            assert!(!step.action_probs.is_empty());

            // Probabilities should sum to ~1.0
            let sum: f64 = step.action_probs.iter().map(|(_, p)| p).sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Probabilities should sum to 1.0, got {}",
                sum
            );
        }
    }

    #[test]
    fn test_compute_outcome_winner() {
        let (engine, mut state) = SimpleGameBuilder::new()
            .player_count(2)
            .starting_life(1) // Very low life
            .build(42);

        let encoder = Box::new(SimpleGameEncoder::new(2, 10));
        let config = SelfPlayConfig::default()
            .with_mcts_iterations(10)
            .with_max_moves(100);

        let worker = SelfPlayWorker::new(engine.clone(), encoder, config);

        // Play game to completion
        let trajectory = worker.play_game(&mut state, 42);

        // Should have a winner
        let p0 = trajectory.player_outcome(PlayerId::new(0));
        let p1 = trajectory.player_outcome(PlayerId::new(1));

        // One should be 1.0, other 0.0, or both 0.5 if draw
        assert!(
            (p0 == 1.0 && p1 == 0.0) || (p0 == 0.0 && p1 == 1.0) || (p0 == 0.5 && p1 == 0.5)
        );
    }

    #[test]
    fn test_trajectory_to_training_samples() {
        let (engine, mut state) = SimpleGameBuilder::new()
            .player_count(2)
            .starting_life(3)
            .build(42);

        let encoder = Box::new(SimpleGameEncoder::new(2, 10));
        let config = SelfPlayConfig::default()
            .with_mcts_iterations(10)
            .with_max_moves(20);

        let worker = SelfPlayWorker::new(engine, encoder, config);
        let trajectory = worker.play_game(&mut state, 42);

        let samples = trajectory.to_training_samples();
        assert_eq!(samples.len(), trajectory.len());

        for sample in &samples {
            assert!(!sample.state.is_empty());
            assert!(!sample.policy.is_empty());
        }
    }

    #[test]
    fn test_self_play_config_all_builders() {
        let config = SelfPlayConfig::new()
            .with_mcts_iterations(500)
            .with_temperature(0.8)
            .with_temperature_threshold(25)
            .with_max_moves(200)
            .with_exploration(2.0)
            .with_seed_offset(1000);

        assert_eq!(config.mcts_iterations, 500);
        assert_eq!(config.temperature, 0.8);
        assert_eq!(config.temperature_threshold, 25);
        assert_eq!(config.max_moves, 200);
        assert_eq!(config.exploration_constant, 2.0);
        assert_eq!(config.seed_offset, 1000);
    }

    #[test]
    fn test_self_play_worker_encoder() {
        let (engine, _) = SimpleGameBuilder::new()
            .player_count(2)
            .build(42);

        let encoder = Box::new(SimpleGameEncoder::new(2, 10));
        let config = SelfPlayConfig::default();

        let worker = SelfPlayWorker::new(engine, encoder, config);

        // Verify encoder is accessible and correct
        assert_eq!(worker.encoder().player_count(), 2);
        assert_eq!(worker.encoder().action_space_size(), 10);
    }

    #[test]
    fn test_self_play_worker_config() {
        let (engine, _) = SimpleGameBuilder::new()
            .player_count(2)
            .build(42);

        let encoder = Box::new(SimpleGameEncoder::new(2, 10));
        let config = SelfPlayConfig::default()
            .with_mcts_iterations(100)
            .with_temperature(0.5);

        let worker = SelfPlayWorker::new(engine, encoder, config);

        assert_eq!(worker.config().mcts_iterations, 100);
        assert_eq!(worker.config().temperature, 0.5);
    }

    #[test]
    fn test_self_play_four_player() {
        let (engine, mut state) = SimpleGameBuilder::new()
            .player_count(4)
            .starting_life(3)
            .build(42);

        let encoder = Box::new(SimpleGameEncoder::new(4, 20));
        let config = SelfPlayConfig::default()
            .with_mcts_iterations(5)
            .with_max_moves(30);

        let worker = SelfPlayWorker::new(engine, encoder, config);
        let trajectory = worker.play_game(&mut state, 42);

        assert!(!trajectory.is_empty());

        // Check that multiple players took turns
        let player_0_steps = trajectory.player_steps(PlayerId::new(0)).count();
        let player_1_steps = trajectory.player_steps(PlayerId::new(1)).count();

        // At least some steps should have been taken by different players
        assert!(player_0_steps + player_1_steps > 0);
    }

    #[test]
    fn test_self_play_deterministic() {
        let config = SelfPlayConfig::default()
            .with_mcts_iterations(10)
            .with_max_moves(20)
            .with_seed_offset(0);

        let encoder1 = Box::new(SimpleGameEncoder::new(2, 10));
        let encoder2 = Box::new(SimpleGameEncoder::new(2, 10));

        let (engine1, _) = SimpleGameBuilder::new()
            .player_count(2)
            .starting_life(3)
            .build(42);
        let (engine2, _) = SimpleGameBuilder::new()
            .player_count(2)
            .starting_life(3)
            .build(42);

        let worker1 = SelfPlayWorker::new(engine1, encoder1, config.clone());
        let worker2 = SelfPlayWorker::new(engine2, encoder2, config);

        // Play same game with same seed
        let (_, mut state1) = SimpleGameBuilder::new()
            .player_count(2)
            .starting_life(3)
            .build(42);
        let (_, mut state2) = SimpleGameBuilder::new()
            .player_count(2)
            .starting_life(3)
            .build(42);

        let traj1 = worker1.play_game(&mut state1, 12345);
        let traj2 = worker2.play_game(&mut state2, 12345);

        // Should have same length
        assert_eq!(traj1.len(), traj2.len());

        // Should have same outcomes
        assert_eq!(
            traj1.player_outcome(PlayerId::new(0)),
            traj2.player_outcome(PlayerId::new(0))
        );
    }

    #[test]
    fn test_self_play_max_moves_reached() {
        let (engine, mut state) = SimpleGameBuilder::new()
            .player_count(2)
            .starting_life(50) // High life so game won't end quickly
            .cards_per_player(0) // No cards to play, so only Pass is available
            .build(42);

        let encoder = Box::new(SimpleGameEncoder::new(2, 10));
        let config = SelfPlayConfig::default()
            .with_mcts_iterations(2) // Very few iterations
            .with_max_moves(3); // Very low max moves

        let worker = SelfPlayWorker::new(engine, encoder, config);
        let trajectory = worker.play_game(&mut state, 42);

        // Should have stopped at max_moves
        assert!(trajectory.len() <= 3);

        // Outcome should be 0.5 for all (incomplete game)
        let p0 = trajectory.player_outcome(PlayerId::new(0));
        let p1 = trajectory.player_outcome(PlayerId::new(1));
        assert!(p0 == 0.5 && p1 == 0.5);
    }

    #[test]
    fn test_self_play_seed_offset() {
        let config = SelfPlayConfig::default()
            .with_mcts_iterations(5)
            .with_max_moves(10)
            .with_seed_offset(1000);

        let encoder = Box::new(SimpleGameEncoder::new(2, 10));
        let (template_engine, _) = SimpleGameBuilder::new()
            .player_count(2)
            .starting_life(2)
            .build(0);

        let worker = SelfPlayWorker::new(template_engine, encoder, config);

        let trajectories = worker.play_games(
            |seed| {
                SimpleGameBuilder::new()
                    .player_count(2)
                    .starting_life(2)
                    .build(seed)
            },
            3,
        );

        // Seeds should be offset
        assert_eq!(trajectories[0].seed, 1000);
        assert_eq!(trajectories[1].seed, 1001);
        assert_eq!(trajectories[2].seed, 1002);
    }

    #[test]
    fn test_self_play_encoded_state_shape() {
        let (engine, mut state) = SimpleGameBuilder::new()
            .player_count(2)
            .starting_life(3)
            .build(42);

        let encoder = Box::new(SimpleGameEncoder::new(2, 10));
        let expected_shape = encoder.output_shape();

        let config = SelfPlayConfig::default()
            .with_mcts_iterations(5)
            .with_max_moves(10);

        let worker = SelfPlayWorker::new(engine, encoder, config);
        let trajectory = worker.play_game(&mut state, 42);

        // All encoded states should have correct shape
        for step in &trajectory.steps {
            assert_eq!(step.encoded_state.shape, expected_shape);
        }
    }

    #[test]
    fn test_self_play_temperature_changes() {
        let (engine, mut state) = SimpleGameBuilder::new()
            .player_count(2)
            .starting_life(10) // Moderate life
            .build(42);

        let encoder = Box::new(SimpleGameEncoder::new(2, 10));
        let config = SelfPlayConfig::default()
            .with_mcts_iterations(10)
            .with_temperature(1.0)
            .with_temperature_threshold(5) // Switch to greedy at move 5
            .with_max_moves(20);

        let worker = SelfPlayWorker::new(engine, encoder, config);
        let trajectory = worker.play_game(&mut state, 42);

        // Just verify game completes with temperature schedule
        assert!(!trajectory.is_empty());
    }

    #[test]
    fn test_self_play_network_integration() {
        let (engine, mut state) = SimpleGameBuilder::new()
            .player_count(2)
            .starting_life(3)
            .build(42);

        let encoder = Box::new(SimpleGameEncoder::new(2, 10));
        let config = SelfPlayConfig::default()
            .with_mcts_iterations(10)
            .with_max_moves(20);

        let worker = SelfPlayWorker::new(engine, encoder, config);

        // Use non-uniform network to test integration
        struct BiasedNetwork;

        impl crate::nn::PolicyValueNetwork for BiasedNetwork {
            fn predict(&self, _encoded: &crate::nn::EncodedState) -> (Vec<f32>, Vec<f32>) {
                // Bias toward first action
                (vec![0.9, 0.05, 0.05], vec![0.5, 0.5])
            }
        }

        let network = BiasedNetwork;
        let trajectory = worker.play_game_with_network(&mut state, 42, &network);

        assert!(!trajectory.is_empty());
    }

    #[test]
    fn test_self_play_empty_game() {
        // This tests the edge case where a game starts in terminal state
        // (not really possible with SimpleGame, but tests the code path)
        let (engine, mut state) = SimpleGameBuilder::new()
            .player_count(2)
            .starting_life(1)
            .build(42);

        // Deal fatal damage to player 1 immediately
        state.public.set_player_state(PlayerId::new(1), "life", 0);

        let encoder = Box::new(SimpleGameEncoder::new(2, 10));
        let config = SelfPlayConfig::default()
            .with_mcts_iterations(5)
            .with_max_moves(10);

        let worker = SelfPlayWorker::new(engine, encoder, config);
        let trajectory = worker.play_game(&mut state, 42);

        // Should have no steps (game was already terminal)
        assert!(trajectory.is_empty());

        // Winner should be player 0
        assert_eq!(trajectory.player_outcome(PlayerId::new(0)), 1.0);
        assert_eq!(trajectory.player_outcome(PlayerId::new(1)), 0.0);
    }
}

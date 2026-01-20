//! Trajectory and experience buffer for training data collection.
//!
//! A trajectory records a complete game from MCTS self-play, capturing:
//! - Encoded states at each decision point
//! - MCTS action probabilities (the "target" policy)
//! - Actions actually taken
//! - Final game outcome for value targets

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::core::{Action, PlayerId, PlayerMap};
use crate::nn::EncodedState;

/// A single step in a trajectory.
///
/// Captures the state, MCTS policy, and action taken at one decision point.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Step {
    /// Encoded game state from the acting player's perspective.
    pub encoded_state: EncodedState,

    /// MCTS action probabilities (target for policy network).
    /// These are the visit counts normalized to probabilities.
    pub action_probs: Vec<(Action, f64)>,

    /// The action that was actually taken.
    pub action_taken: Action,

    /// The player who made this decision.
    pub player: PlayerId,

    /// Move number in the game (0-indexed).
    pub move_number: usize,
}

impl Step {
    /// Create a new step.
    pub fn new(
        encoded_state: EncodedState,
        action_probs: Vec<(Action, f64)>,
        action_taken: Action,
        player: PlayerId,
        move_number: usize,
    ) -> Self {
        Self {
            encoded_state,
            action_probs,
            action_taken,
            player,
            move_number,
        }
    }

    /// Get the probability assigned to the taken action.
    pub fn taken_action_prob(&self) -> f64 {
        self.action_probs
            .iter()
            .find(|(a, _)| a == &self.action_taken)
            .map(|(_, p)| *p)
            .unwrap_or(0.0)
    }
}

/// A complete game trajectory from self-play.
///
/// Contains all decision points and the final outcome,
/// which provides value targets for training.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Trajectory {
    /// All steps in the game.
    pub steps: Vec<Step>,

    /// Final outcome (reward for each player).
    /// Typically 1.0 for winner, 0.0 for losers, 0.5 for draw.
    pub outcome: PlayerMap<f64>,

    /// Total number of moves in the game.
    pub game_length: usize,

    /// Random seed used for this game.
    pub seed: u64,
}

impl Trajectory {
    /// Create a new trajectory.
    pub fn new(seed: u64, player_count: usize) -> Self {
        Self {
            steps: Vec::new(),
            outcome: PlayerMap::with_value(player_count, 0.0),
            game_length: 0,
            seed,
        }
    }

    /// Add a step to the trajectory.
    pub fn push(&mut self, step: Step) {
        self.steps.push(step);
        self.game_length += 1;
    }

    /// Set the final outcome.
    pub fn set_outcome(&mut self, outcome: PlayerMap<f64>) {
        self.outcome = outcome;
    }

    /// Get the number of steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Check if trajectory is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Get steps for a specific player.
    pub fn player_steps(&self, player: PlayerId) -> impl Iterator<Item = &Step> {
        self.steps.iter().filter(move |s| s.player == player)
    }

    /// Get the outcome for a specific player.
    pub fn player_outcome(&self, player: PlayerId) -> f64 {
        self.outcome[player]
    }

    /// Convert to training samples.
    ///
    /// Each sample contains:
    /// - Encoded state
    /// - Target policy (MCTS probabilities)
    /// - Target value (game outcome from player's perspective)
    pub fn to_training_samples(&self) -> Vec<TrainingSample> {
        self.steps
            .iter()
            .map(|step| TrainingSample {
                state: step.encoded_state.clone(),
                policy: step.action_probs.iter().map(|(_, p)| *p as f32).collect(),
                value: self.outcome[step.player] as f32,
                player: step.player,
            })
            .collect()
    }
}

/// A single training sample extracted from a trajectory.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingSample {
    /// Encoded game state.
    pub state: EncodedState,

    /// Target policy (MCTS visit distribution).
    pub policy: Vec<f32>,

    /// Target value (game outcome).
    pub value: f32,

    /// Player whose perspective this is from.
    pub player: PlayerId,
}

/// Buffer for storing trajectories during training.
///
/// Uses a FIFO strategy: when full, oldest trajectories are removed.
#[derive(Clone, Debug)]
pub struct ExperienceBuffer {
    trajectories: VecDeque<Trajectory>,
    max_trajectories: usize,
}

impl ExperienceBuffer {
    /// Create a new experience buffer.
    pub fn new(max_trajectories: usize) -> Self {
        Self {
            trajectories: VecDeque::with_capacity(max_trajectories),
            max_trajectories,
        }
    }

    /// Add a trajectory to the buffer.
    ///
    /// If the buffer is full, the oldest trajectory is removed.
    pub fn push(&mut self, trajectory: Trajectory) {
        if self.trajectories.len() >= self.max_trajectories {
            self.trajectories.pop_front();
        }
        self.trajectories.push_back(trajectory);
    }

    /// Get the number of trajectories in the buffer.
    pub fn len(&self) -> usize {
        self.trajectories.len()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.trajectories.is_empty()
    }

    /// Get the maximum capacity.
    pub fn capacity(&self) -> usize {
        self.max_trajectories
    }

    /// Clear all trajectories.
    pub fn clear(&mut self) {
        self.trajectories.clear();
    }

    /// Get an iterator over trajectories.
    pub fn iter(&self) -> impl Iterator<Item = &Trajectory> {
        self.trajectories.iter()
    }

    /// Get the total number of steps across all trajectories.
    pub fn total_steps(&self) -> usize {
        self.trajectories.iter().map(|t| t.len()).sum()
    }

    /// Extract all training samples from the buffer.
    pub fn to_training_samples(&self) -> Vec<TrainingSample> {
        self.trajectories
            .iter()
            .flat_map(|t| t.to_training_samples())
            .collect()
    }

    /// Sample a random batch of training samples.
    ///
    /// Uses the provided RNG seed for reproducibility.
    pub fn sample_batch(&self, batch_size: usize, seed: u64) -> Vec<TrainingSample> {
        use crate::core::GameRng;

        let all_samples = self.to_training_samples();
        if all_samples.is_empty() || batch_size == 0 {
            return vec![];
        }

        let mut rng = GameRng::new(seed);

        // Fisher-Yates shuffle first `batch_size` elements
        let mut indices: Vec<usize> = (0..all_samples.len()).collect();
        let n = indices.len();
        let limit = batch_size.min(n);

        for i in 0..limit {
            let j = i + rng.gen_range_usize(0..n - i);
            indices.swap(i, j);
        }

        indices
            .into_iter()
            .take(batch_size)
            .map(|i| all_samples[i].clone())
            .collect()
    }
}

impl Default for ExperienceBuffer {
    fn default() -> Self {
        Self::new(10000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TemplateId;
    use crate::nn::EncodedState;

    fn make_test_step(player: u8, move_num: usize) -> Step {
        let action = Action::new(TemplateId::new(0));
        Step::new(
            EncodedState::zeros(vec![10]),
            vec![(action.clone(), 0.5), (Action::new(TemplateId::new(1)), 0.5)],
            action,
            PlayerId::new(player),
            move_num,
        )
    }

    #[test]
    fn test_step_creation() {
        let step = make_test_step(0, 5);
        assert_eq!(step.player, PlayerId::new(0));
        assert_eq!(step.move_number, 5);
        assert_eq!(step.action_probs.len(), 2);
    }

    #[test]
    fn test_step_taken_action_prob() {
        let action = Action::new(TemplateId::new(0));
        let step = Step::new(
            EncodedState::zeros(vec![10]),
            vec![(action.clone(), 0.7), (Action::new(TemplateId::new(1)), 0.3)],
            action,
            PlayerId::new(0),
            0,
        );

        assert!((step.taken_action_prob() - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_trajectory_creation() {
        let mut traj = Trajectory::new(42, 2);
        assert!(traj.is_empty());
        assert_eq!(traj.game_length, 0);

        traj.push(make_test_step(0, 0));
        traj.push(make_test_step(1, 1));

        assert_eq!(traj.len(), 2);
        assert_eq!(traj.game_length, 2);
    }

    #[test]
    fn test_trajectory_outcome() {
        let mut traj = Trajectory::new(42, 2);
        traj.push(make_test_step(0, 0));

        let mut outcome = PlayerMap::with_value(2, 0.0);
        outcome[PlayerId::new(0)] = 1.0;
        traj.set_outcome(outcome);

        assert_eq!(traj.player_outcome(PlayerId::new(0)), 1.0);
        assert_eq!(traj.player_outcome(PlayerId::new(1)), 0.0);
    }

    #[test]
    fn test_trajectory_player_steps() {
        let mut traj = Trajectory::new(42, 2);
        traj.push(make_test_step(0, 0));
        traj.push(make_test_step(1, 1));
        traj.push(make_test_step(0, 2));
        traj.push(make_test_step(1, 3));

        let p0_steps: Vec<_> = traj.player_steps(PlayerId::new(0)).collect();
        let p1_steps: Vec<_> = traj.player_steps(PlayerId::new(1)).collect();

        assert_eq!(p0_steps.len(), 2);
        assert_eq!(p1_steps.len(), 2);
    }

    #[test]
    fn test_trajectory_to_training_samples() {
        let mut traj = Trajectory::new(42, 2);
        traj.push(make_test_step(0, 0));
        traj.push(make_test_step(1, 1));

        let mut outcome = PlayerMap::with_value(2, 0.0);
        outcome[PlayerId::new(0)] = 1.0;
        traj.set_outcome(outcome);

        let samples = traj.to_training_samples();
        assert_eq!(samples.len(), 2);

        // Player 0's sample should have value 1.0
        assert_eq!(samples[0].value, 1.0);
        // Player 1's sample should have value 0.0
        assert_eq!(samples[1].value, 0.0);
    }

    #[test]
    fn test_experience_buffer_capacity() {
        let mut buffer = ExperienceBuffer::new(3);

        buffer.push(Trajectory::new(1, 2));
        buffer.push(Trajectory::new(2, 2));
        buffer.push(Trajectory::new(3, 2));
        assert_eq!(buffer.len(), 3);

        // Adding a 4th should evict the oldest
        buffer.push(Trajectory::new(4, 2));
        assert_eq!(buffer.len(), 3);

        // Check that trajectory with seed 1 was evicted
        let seeds: Vec<_> = buffer.iter().map(|t| t.seed).collect();
        assert!(!seeds.contains(&1));
        assert!(seeds.contains(&2));
        assert!(seeds.contains(&3));
        assert!(seeds.contains(&4));
    }

    #[test]
    fn test_experience_buffer_total_steps() {
        let mut buffer = ExperienceBuffer::new(10);

        let mut traj1 = Trajectory::new(1, 2);
        traj1.push(make_test_step(0, 0));
        traj1.push(make_test_step(1, 1));

        let mut traj2 = Trajectory::new(2, 2);
        traj2.push(make_test_step(0, 0));

        buffer.push(traj1);
        buffer.push(traj2);

        assert_eq!(buffer.total_steps(), 3);
    }

    #[test]
    fn test_experience_buffer_sample_batch() {
        let mut buffer = ExperienceBuffer::new(10);

        let mut traj = Trajectory::new(42, 2);
        for i in 0..10 {
            traj.push(make_test_step((i % 2) as u8, i));
        }
        buffer.push(traj);

        let batch = buffer.sample_batch(5, 123);
        assert_eq!(batch.len(), 5);

        // With different seed, should get different order (usually)
        let batch2 = buffer.sample_batch(5, 456);
        assert_eq!(batch2.len(), 5);
    }

    #[test]
    fn test_experience_buffer_sample_batch_larger_than_available() {
        let mut buffer = ExperienceBuffer::new(10);

        let mut traj = Trajectory::new(42, 2);
        traj.push(make_test_step(0, 0));
        traj.push(make_test_step(1, 1));
        buffer.push(traj);

        // Request more samples than available
        let batch = buffer.sample_batch(100, 123);
        assert_eq!(batch.len(), 2); // Only 2 samples available
    }

    #[test]
    fn test_trajectory_serialization() {
        let mut traj = Trajectory::new(42, 2);
        traj.push(make_test_step(0, 0));

        let mut outcome = PlayerMap::with_value(2, 0.0);
        outcome[PlayerId::new(0)] = 1.0;
        traj.set_outcome(outcome);

        let json = serde_json::to_string(&traj).unwrap();
        let deserialized: Trajectory = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.seed, 42);
        assert_eq!(deserialized.len(), 1);
        assert_eq!(deserialized.player_outcome(PlayerId::new(0)), 1.0);
    }

    #[test]
    fn test_step_action_not_found() {
        let action1 = Action::new(TemplateId::new(0));
        let action2 = Action::new(TemplateId::new(1));
        let action3 = Action::new(TemplateId::new(2));

        let step = Step::new(
            EncodedState::zeros(vec![10]),
            vec![(action1, 0.5), (action2, 0.5)],
            action3, // Taken action not in probs
            PlayerId::new(0),
            0,
        );

        // Should return 0.0 when taken action not found
        assert_eq!(step.taken_action_prob(), 0.0);
    }

    #[test]
    fn test_step_fields() {
        let action = Action::new(TemplateId::new(5));
        let encoded = EncodedState::new(vec![1.0, 2.0, 3.0], vec![3]);

        let step = Step::new(
            encoded.clone(),
            vec![(action.clone(), 1.0)],
            action.clone(),
            PlayerId::new(2),
            10,
        );

        assert_eq!(step.player, PlayerId::new(2));
        assert_eq!(step.move_number, 10);
        assert_eq!(step.action_taken, action);
        assert_eq!(step.encoded_state.tensor, encoded.tensor);
        assert_eq!(step.action_probs.len(), 1);
    }

    #[test]
    fn test_training_sample_fields() {
        let mut traj = Trajectory::new(42, 2);
        traj.push(make_test_step(0, 0));

        let mut outcome = PlayerMap::with_value(2, 0.0);
        outcome[PlayerId::new(0)] = 0.75;
        traj.set_outcome(outcome);

        let samples = traj.to_training_samples();
        assert_eq!(samples.len(), 1);

        let sample = &samples[0];
        assert_eq!(sample.player, PlayerId::new(0));
        assert_eq!(sample.value, 0.75);
        assert!(!sample.state.is_empty());
        assert_eq!(sample.policy.len(), 2); // Two action probs
    }

    #[test]
    fn test_experience_buffer_clear() {
        let mut buffer = ExperienceBuffer::new(10);

        buffer.push(Trajectory::new(1, 2));
        buffer.push(Trajectory::new(2, 2));
        buffer.push(Trajectory::new(3, 2));

        assert_eq!(buffer.len(), 3);

        buffer.clear();

        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_experience_buffer_default() {
        let buffer = ExperienceBuffer::default();
        assert_eq!(buffer.capacity(), 10000);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_experience_buffer_capacity_method() {
        let buffer = ExperienceBuffer::new(50);
        assert_eq!(buffer.capacity(), 50);
    }

    #[test]
    fn test_experience_buffer_empty_operations() {
        let buffer = ExperienceBuffer::new(10);

        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.total_steps(), 0);
        assert!(buffer.to_training_samples().is_empty());
        assert!(buffer.sample_batch(10, 42).is_empty());
    }

    #[test]
    fn test_experience_buffer_sample_batch_zero_size() {
        let mut buffer = ExperienceBuffer::new(10);

        let mut traj = Trajectory::new(42, 2);
        traj.push(make_test_step(0, 0));
        buffer.push(traj);

        let batch = buffer.sample_batch(0, 123);
        assert!(batch.is_empty());
    }

    #[test]
    fn test_experience_buffer_sample_deterministic() {
        let mut buffer = ExperienceBuffer::new(10);

        let mut traj = Trajectory::new(42, 2);
        for i in 0..10 {
            traj.push(make_test_step((i % 2) as u8, i));
        }
        buffer.push(traj);

        // Same seed should give same batch
        let batch1 = buffer.sample_batch(5, 12345);
        let batch2 = buffer.sample_batch(5, 12345);

        assert_eq!(batch1.len(), batch2.len());
        for (s1, s2) in batch1.iter().zip(batch2.iter()) {
            assert_eq!(s1.player, s2.player);
        }
    }

    #[test]
    fn test_trajectory_four_player() {
        let mut traj = Trajectory::new(42, 4);

        for i in 0..8 {
            let step = Step::new(
                EncodedState::zeros(vec![20]),
                vec![(Action::new(TemplateId::new(0)), 1.0)],
                Action::new(TemplateId::new(0)),
                PlayerId::new((i % 4) as u8),
                i,
            );
            traj.push(step);
        }

        // Each player should have 2 steps
        for p in 0..4 {
            let count = traj.player_steps(PlayerId::new(p)).count();
            assert_eq!(count, 2);
        }
    }

    #[test]
    fn test_trajectory_draw_outcome() {
        let mut traj = Trajectory::new(42, 2);
        traj.push(make_test_step(0, 0));

        let mut outcome = PlayerMap::with_value(2, 0.0);
        outcome[PlayerId::new(0)] = 0.5;
        outcome[PlayerId::new(1)] = 0.5;
        traj.set_outcome(outcome);

        assert_eq!(traj.player_outcome(PlayerId::new(0)), 0.5);
        assert_eq!(traj.player_outcome(PlayerId::new(1)), 0.5);
    }

    #[test]
    fn test_step_serialization() {
        let step = make_test_step(1, 5);

        let json = serde_json::to_string(&step).unwrap();
        let deserialized: Step = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.player, PlayerId::new(1));
        assert_eq!(deserialized.move_number, 5);
        assert_eq!(deserialized.action_probs.len(), 2);
    }

    #[test]
    fn test_training_sample_serialization() {
        let sample = TrainingSample {
            state: EncodedState::zeros(vec![10]),
            policy: vec![0.3, 0.7],
            value: 0.8,
            player: PlayerId::new(0),
        };

        let json = serde_json::to_string(&sample).unwrap();
        let deserialized: TrainingSample = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.value, 0.8);
        assert_eq!(deserialized.policy, vec![0.3, 0.7]);
        assert_eq!(deserialized.player, PlayerId::new(0));
    }

    #[test]
    fn test_experience_buffer_iter() {
        let mut buffer = ExperienceBuffer::new(10);

        buffer.push(Trajectory::new(1, 2));
        buffer.push(Trajectory::new(2, 2));
        buffer.push(Trajectory::new(3, 2));

        let seeds: Vec<_> = buffer.iter().map(|t| t.seed).collect();
        assert_eq!(seeds, vec![1, 2, 3]);
    }

    #[test]
    fn test_trajectory_to_samples_preserves_order() {
        let mut traj = Trajectory::new(42, 2);
        traj.push(make_test_step(0, 0));
        traj.push(make_test_step(1, 1));
        traj.push(make_test_step(0, 2));

        let samples = traj.to_training_samples();

        assert_eq!(samples[0].player, PlayerId::new(0));
        assert_eq!(samples[1].player, PlayerId::new(1));
        assert_eq!(samples[2].player, PlayerId::new(0));
    }

    #[test]
    fn test_experience_buffer_fifo_order() {
        let mut buffer = ExperienceBuffer::new(3);

        // Fill buffer
        buffer.push(Trajectory::new(1, 2));
        buffer.push(Trajectory::new(2, 2));
        buffer.push(Trajectory::new(3, 2));

        // Add more, triggering eviction
        buffer.push(Trajectory::new(4, 2));
        buffer.push(Trajectory::new(5, 2));

        // Should have 3, 4, 5 (oldest evicted first)
        let seeds: Vec<_> = buffer.iter().map(|t| t.seed).collect();
        assert_eq!(seeds, vec![3, 4, 5]);
    }
}

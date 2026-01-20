//! Integration tests for neural network and training infrastructure.

use rust_ccg::core::PlayerId;
use rust_ccg::games::simple::SimpleGameBuilder;
use rust_ccg::mcts::{MCTSConfig, MCTSSearch};
use rust_ccg::nn::{
    EncodedState, PolicyNetwork, PolicyValueNetwork, SimpleGameEncoder, StateEncoder, UniformPolicy,
    UniformPolicyZeroValue, ZeroEncoder,
};
use rust_ccg::training::{ExperienceBuffer, SelfPlayConfig, SelfPlayWorker, Step, Trajectory};

// =============================================================================
// Encoder Tests
// =============================================================================

#[test]
fn test_simple_game_encoder_output_shape() {
    let encoder = SimpleGameEncoder::new(2, 10);
    let shape = encoder.output_shape();

    // 5 features per player * 2 players = 10
    assert_eq!(shape, vec![10]);
}

#[test]
fn test_simple_game_encoder_four_players() {
    let encoder = SimpleGameEncoder::new(4, 20);
    let shape = encoder.output_shape();

    // 5 features per player * 4 players = 20
    assert_eq!(shape, vec![20]);
}

#[test]
fn test_encoder_encodes_state() {
    let (_, state) = SimpleGameBuilder::new()
        .player_count(2)
        .starting_life(20)
        .build(42);

    let encoder = SimpleGameEncoder::new(2, 10).with_max_life(20.0);
    let encoded = encoder.encode(&state, PlayerId::new(0));

    // Check tensor is correct size
    assert_eq!(encoded.len(), 10);
    assert_eq!(encoded.shape, vec![10]);

    // Player 0's life should be 20/20 = 1.0
    assert!((encoded.tensor[0] - 1.0).abs() < 0.01);
}

#[test]
fn test_encoder_perspective() {
    let (_, state) = SimpleGameBuilder::new()
        .player_count(2)
        .build(42);

    let encoder = SimpleGameEncoder::new(2, 10);

    // Encode from player 0's perspective
    let encoded_p0 = encoder.encode(&state, PlayerId::new(0));
    // Encode from player 1's perspective
    let encoded_p1 = encoder.encode(&state, PlayerId::new(1));

    // Perspective indicators should differ
    // Player 0's perspective flag at index 4
    assert_eq!(encoded_p0.tensor[4], 1.0);
    assert_eq!(encoded_p0.tensor[9], 0.0);

    // Player 1's perspective flag at index 9
    assert_eq!(encoded_p1.tensor[4], 0.0);
    assert_eq!(encoded_p1.tensor[9], 1.0);
}

#[test]
fn test_zero_encoder() {
    let (_, state) = SimpleGameBuilder::new()
        .player_count(2)
        .build(42);

    let encoder = ZeroEncoder::new(vec![3, 4], 10, 2);
    let encoded = encoder.encode(&state, PlayerId::new(0));

    assert_eq!(encoded.len(), 12);
    assert_eq!(encoded.shape, vec![3, 4]);
    assert!(encoded.tensor.iter().all(|&v| v == 0.0));
}

// =============================================================================
// Network Trait Tests
// =============================================================================

#[test]
fn test_uniform_policy() {
    let policy = UniformPolicy::new(5);
    let state = EncodedState::zeros(vec![10]);
    let probs = policy.predict(&state);

    assert_eq!(probs.len(), 5);
    assert!((probs.iter().sum::<f32>() - 1.0).abs() < 0.001);
}

#[test]
fn test_uniform_policy_zero_value() {
    let network = UniformPolicyZeroValue::new(10, 2);
    let state = EncodedState::zeros(vec![20]);
    let (policy, value) = network.predict(&state);

    assert_eq!(policy.len(), 10);
    assert_eq!(value.len(), 2);
    assert!((policy.iter().sum::<f32>() - 1.0).abs() < 0.001);
    assert!(value.iter().all(|&v| v == 0.0));
}

// =============================================================================
// Trajectory Tests
// =============================================================================

#[test]
fn test_trajectory_creation() {
    let traj = Trajectory::new(42, 2);
    assert!(traj.is_empty());
    assert_eq!(traj.seed, 42);
}

#[test]
fn test_trajectory_with_steps() {
    let mut traj = Trajectory::new(42, 2);

    let action = rust_ccg::core::Action::new(rust_ccg::core::TemplateId::new(0));
    let step = Step::new(
        EncodedState::zeros(vec![10]),
        vec![(action.clone(), 1.0)],
        action,
        PlayerId::new(0),
        0,
    );

    traj.push(step);
    assert_eq!(traj.len(), 1);
}

#[test]
fn test_trajectory_to_samples() {
    let mut traj = Trajectory::new(42, 2);

    let action = rust_ccg::core::Action::new(rust_ccg::core::TemplateId::new(0));
    let step = Step::new(
        EncodedState::zeros(vec![10]),
        vec![(action.clone(), 1.0)],
        action,
        PlayerId::new(0),
        0,
    );

    traj.push(step);

    // Set outcome
    let mut outcome = rust_ccg::core::PlayerMap::with_value(2, 0.0);
    outcome[PlayerId::new(0)] = 1.0;
    traj.set_outcome(outcome);

    let samples = traj.to_training_samples();
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].value, 1.0);
}

// =============================================================================
// Experience Buffer Tests
// =============================================================================

#[test]
fn test_experience_buffer_capacity() {
    let mut buffer = ExperienceBuffer::new(3);

    for i in 0..5 {
        buffer.push(Trajectory::new(i, 2));
    }

    // Should only have 3 trajectories
    assert_eq!(buffer.len(), 3);

    // Should have the most recent ones (2, 3, 4)
    let seeds: Vec<_> = buffer.iter().map(|t| t.seed).collect();
    assert!(seeds.contains(&2));
    assert!(seeds.contains(&3));
    assert!(seeds.contains(&4));
}

#[test]
fn test_experience_buffer_sampling() {
    let mut buffer = ExperienceBuffer::new(10);

    let action = rust_ccg::core::Action::new(rust_ccg::core::TemplateId::new(0));
    let mut traj = Trajectory::new(42, 2);

    for i in 0..5 {
        let step = Step::new(
            EncodedState::zeros(vec![10]),
            vec![(action.clone(), 1.0)],
            action.clone(),
            PlayerId::new(0),
            i,
        );
        traj.push(step);
    }

    buffer.push(traj);

    let samples = buffer.sample_batch(3, 123);
    assert_eq!(samples.len(), 3);
}

// =============================================================================
// Self-Play Tests
// =============================================================================

#[test]
fn test_self_play_config() {
    let config = SelfPlayConfig::default()
        .with_mcts_iterations(100)
        .with_temperature(0.5)
        .with_temperature_threshold(20);

    assert_eq!(config.mcts_iterations, 100);
    assert_eq!(config.temperature, 0.5);
    assert_eq!(config.temperature_threshold, 20);
}

#[test]
fn test_self_play_generates_trajectory() {
    let (engine, mut state) = SimpleGameBuilder::new()
        .player_count(2)
        .starting_life(3) // Low life for quick game
        .build(42);

    let encoder = Box::new(SimpleGameEncoder::new(2, 10));
    let config = SelfPlayConfig::default()
        .with_mcts_iterations(10)
        .with_max_moves(30);

    let worker = SelfPlayWorker::new(engine, encoder, config);
    let trajectory = worker.play_game(&mut state, 42);

    // Should have recorded some steps
    assert!(!trajectory.is_empty());

    // Each step should have valid data
    for step in &trajectory.steps {
        assert!(!step.encoded_state.is_empty());
        assert!(!step.action_probs.is_empty());
    }
}

#[test]
fn test_self_play_with_network() {
    let (engine, mut state) = SimpleGameBuilder::new()
        .player_count(2)
        .starting_life(3)
        .build(42);

    let encoder = Box::new(SimpleGameEncoder::new(2, 10));
    let config = SelfPlayConfig::default()
        .with_mcts_iterations(10)
        .with_max_moves(30);

    let worker = SelfPlayWorker::new(engine, encoder, config);
    let network = UniformPolicyZeroValue::new(10, 2);

    let trajectory = worker.play_game_with_network(&mut state, 42, &network);

    assert!(!trajectory.is_empty());
}

#[test]
fn test_self_play_multiple_games() {
    let config = SelfPlayConfig::default()
        .with_mcts_iterations(5)
        .with_max_moves(20);

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

    assert_eq!(trajectories.len(), 3);

    // Each game should have produced a trajectory
    for traj in &trajectories {
        assert!(!traj.is_empty());
    }
}

// =============================================================================
// MCTS Integration Tests
// =============================================================================

#[test]
fn test_mcts_set_root_priors() {
    let (game, mut state) = SimpleGameBuilder::new()
        .player_count(2)
        .starting_life(20)
        .build(42);

    let config = MCTSConfig::default();
    let mut search = MCTSSearch::new(game, config);

    // Run a small search to initialize the tree
    search.search(&mut state, PlayerId::new(0), 10);

    // Get actions from root
    let actions: Vec<_> = search.action_visits().into_iter().map(|(a, _)| a).collect();

    // Set priors
    let priors: Vec<_> = actions.iter().map(|a| (a.clone(), 0.5f32)).collect();
    let updated = search.set_root_priors(&priors);

    assert_eq!(updated, actions.len());

    // Verify priors were set
    let root_priors = search.root_priors();
    assert!(root_priors.iter().all(|(_, p)| (*p - 0.5).abs() < 0.01));
}

#[test]
fn test_training_pipeline_integration() {
    // Full pipeline: self-play -> trajectory -> buffer -> samples

    // 1. Set up self-play
    let (engine, mut state) = SimpleGameBuilder::new()
        .player_count(2)
        .starting_life(2)
        .build(42);

    let encoder = Box::new(SimpleGameEncoder::new(2, 10));
    let config = SelfPlayConfig::default()
        .with_mcts_iterations(5)
        .with_max_moves(20);

    let worker = SelfPlayWorker::new(engine, encoder, config);

    // 2. Generate trajectory
    let trajectory = worker.play_game(&mut state, 42);
    assert!(!trajectory.is_empty());

    // 3. Add to buffer
    let mut buffer = ExperienceBuffer::new(100);
    buffer.push(trajectory);

    assert_eq!(buffer.len(), 1);

    // 4. Sample batch
    let samples = buffer.sample_batch(5, 123);

    // Verify samples
    for sample in &samples {
        assert!(!sample.state.is_empty());
        assert!(!sample.policy.is_empty());
    }
}

// =============================================================================
// Serialization Tests
// =============================================================================

#[test]
fn test_encoded_state_serialization() {
    let state = EncodedState::new(vec![1.0, 2.0, 3.0], vec![3]);
    let json = serde_json::to_string(&state).unwrap();
    let deserialized: EncodedState = serde_json::from_str(&json).unwrap();

    assert_eq!(state.tensor, deserialized.tensor);
    assert_eq!(state.shape, deserialized.shape);
}

#[test]
fn test_trajectory_serialization() {
    let mut traj = Trajectory::new(42, 2);

    let action = rust_ccg::core::Action::new(rust_ccg::core::TemplateId::new(0));
    let step = Step::new(
        EncodedState::zeros(vec![10]),
        vec![(action.clone(), 1.0)],
        action,
        PlayerId::new(0),
        0,
    );

    traj.push(step);

    let json = serde_json::to_string(&traj).unwrap();
    let deserialized: Trajectory = serde_json::from_str(&json).unwrap();

    assert_eq!(traj.seed, deserialized.seed);
    assert_eq!(traj.len(), deserialized.len());
}

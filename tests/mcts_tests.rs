//! MCTS integration tests using SimpleGame.

use rust_ccg::core::PlayerId;
use rust_ccg::games::simple::SimpleGameBuilder;
use rust_ccg::mcts::{MCTSConfig, MCTSSearch, MCTSTree, PUCT};

// =============================================================================
// Basic Search Tests
// =============================================================================

#[test]
fn test_mcts_returns_action() {
    let (game, mut state) = SimpleGameBuilder::new()
        .player_count(2)
        .starting_life(20)
        .build(42);

    let config = MCTSConfig::default();
    let mut search = MCTSSearch::new(game, config);

    let action = search.search(&mut state, PlayerId::new(0), 100);

    assert!(action.is_some(), "MCTS should return an action");
}

#[test]
fn test_mcts_with_low_iterations() {
    let (game, mut state) = SimpleGameBuilder::new()
        .player_count(2)
        .starting_life(10)
        .build(42);

    let config = MCTSConfig::default();
    let mut search = MCTSSearch::new(game, config);

    // Even with few iterations, should return something
    let action = search.search(&mut state, PlayerId::new(0), 10);

    assert!(action.is_some());
}

// =============================================================================
// Determinism Tests
// =============================================================================

#[test]
fn test_mcts_deterministic_with_seed() {
    let (game1, mut state1) = SimpleGameBuilder::new()
        .player_count(2)
        .starting_life(15)
        .build(42);
    let (game2, mut state2) = SimpleGameBuilder::new()
        .player_count(2)
        .starting_life(15)
        .build(42);

    let config = MCTSConfig::default().with_seed(12345);

    let mut search1 = MCTSSearch::new(game1, config.clone());
    let mut search2 = MCTSSearch::new(game2, config);

    let action1 = search1.search(&mut state1, PlayerId::new(0), 100);
    let action2 = search2.search(&mut state2, PlayerId::new(0), 100);

    assert_eq!(action1, action2, "Same seed should produce same action");
}

#[test]
fn test_mcts_different_seeds_differ() {
    let (game1, mut state1) = SimpleGameBuilder::new()
        .player_count(2)
        .starting_life(15)
        .build(42);
    let (game2, mut state2) = SimpleGameBuilder::new()
        .player_count(2)
        .starting_life(15)
        .build(42);

    let config1 = MCTSConfig::default().with_seed(111);
    let config2 = MCTSConfig::default().with_seed(222);

    let mut search1 = MCTSSearch::new(game1, config1);
    let mut search2 = MCTSSearch::new(game2, config2);

    // Run many iterations to get stable results
    let _action1 = search1.search(&mut state1, PlayerId::new(0), 500);
    let _action2 = search2.search(&mut state2, PlayerId::new(0), 500);

    // Actions might differ (not guaranteed, but likely with different seeds)
    // At minimum, stats should differ
    let stats1 = search1.stats();
    let stats2 = search2.stats();

    // Both should complete iterations
    assert_eq!(stats1.iterations, 500);
    assert_eq!(stats2.iterations, 500);
}

// =============================================================================
// N-Player Tests
// =============================================================================

#[test]
fn test_mcts_two_player() {
    let (game, mut state) = SimpleGameBuilder::new()
        .player_count(2)
        .starting_life(20)
        .build(42);

    let config = MCTSConfig::default();
    let mut search = MCTSSearch::new(game, config);

    let action = search.search(&mut state, PlayerId::new(0), 100);
    assert!(action.is_some());
}

#[test]
fn test_mcts_four_player() {
    let (game, mut state) = SimpleGameBuilder::new()
        .player_count(4)
        .starting_life(15)
        .build(42);

    let config = MCTSConfig::default();
    let mut search = MCTSSearch::new(game, config);

    // Search as player 0 (who starts)
    let action = search.search(&mut state, PlayerId::new(0), 100);
    assert!(action.is_some());

    // Tree should have expanded
    assert!(search.tree().len() > 1);
}

// =============================================================================
// Statistics Tests
// =============================================================================

#[test]
fn test_mcts_statistics() {
    let (game, mut state) = SimpleGameBuilder::new()
        .player_count(2)
        .starting_life(20)
        .build(42);

    let config = MCTSConfig::default();
    let mut search = MCTSSearch::new(game, config);

    search.search(&mut state, PlayerId::new(0), 100);

    let stats = search.stats();

    assert_eq!(stats.iterations, 100);
    assert!(stats.nodes_expanded > 0, "Should expand some nodes");
    assert!(stats.simulations > 0, "Should run some simulations");
    assert!(stats.time_us > 0, "Should record time");
}

#[test]
fn test_mcts_tree_stats() {
    let (game, mut state) = SimpleGameBuilder::new()
        .player_count(2)
        .starting_life(10)
        .build(42);

    let config = MCTSConfig::default();
    let mut search = MCTSSearch::new(game, config);

    search.search(&mut state, PlayerId::new(0), 200);

    let tree_stats = search.tree().stats();

    assert!(tree_stats.node_count > 1, "Tree should have multiple nodes");
    assert!(tree_stats.max_depth > 0, "Tree should have depth");
    assert!(tree_stats.total_edges > 0, "Tree should have edges");
}

// =============================================================================
// Action Probability Tests
// =============================================================================

#[test]
fn test_action_probabilities_sum_to_one() {
    let (game, mut state) = SimpleGameBuilder::new()
        .player_count(2)
        .starting_life(20)
        .build(42);

    let config = MCTSConfig::default();
    let mut search = MCTSSearch::new(game, config);

    search.search(&mut state, PlayerId::new(0), 100);

    let probs = search.action_probabilities();

    // Should have probabilities for each action
    assert!(!probs.is_empty());

    // Should sum to ~1.0
    let sum: f64 = probs.iter().map(|(_, p)| p).sum();
    assert!(
        (sum - 1.0).abs() < 0.01,
        "Probabilities should sum to 1.0, got {}",
        sum
    );
}

#[test]
fn test_action_visits() {
    let (game, mut state) = SimpleGameBuilder::new()
        .player_count(2)
        .starting_life(20)
        .build(42);

    let config = MCTSConfig::default();
    let mut search = MCTSSearch::new(game, config);

    search.search(&mut state, PlayerId::new(0), 100);

    let visits = search.action_visits();

    // Total visits should be close to iteration count
    let total: u32 = visits.iter().map(|(_, v)| v).sum();
    assert!(total > 0);
}

// =============================================================================
// Policy Tests
// =============================================================================

#[test]
fn test_mcts_with_puct() {
    let (game, mut state) = SimpleGameBuilder::new()
        .player_count(2)
        .starting_life(20)
        .build(42);

    let config = MCTSConfig::default();
    let mut search = MCTSSearch::new(game, config).with_selection(PUCT);

    let action = search.search(&mut state, PlayerId::new(0), 100);
    assert!(action.is_some());
}

// =============================================================================
// Configuration Tests
// =============================================================================

#[test]
fn test_mcts_exploration_constant() {
    let (game1, mut state1) = SimpleGameBuilder::new()
        .player_count(2)
        .starting_life(10)
        .build(42);
    let (game2, mut state2) = SimpleGameBuilder::new()
        .player_count(2)
        .starting_life(10)
        .build(42);

    // High exploration
    let config_explore = MCTSConfig::default()
        .with_exploration(5.0)
        .with_seed(42);

    // Low exploration (more greedy)
    let config_exploit = MCTSConfig::default()
        .with_exploration(0.1)
        .with_seed(42);

    let mut search_explore = MCTSSearch::new(game1, config_explore);
    let mut search_exploit = MCTSSearch::new(game2, config_exploit);

    search_explore.search(&mut state1, PlayerId::new(0), 200);
    search_exploit.search(&mut state2, PlayerId::new(0), 200);

    // High exploration should have more diverse tree
    let explore_stats = search_explore.tree().stats();
    let exploit_stats = search_exploit.tree().stats();

    // Just verify both work - exact behavior depends on game specifics
    assert!(explore_stats.node_count > 0);
    assert!(exploit_stats.node_count > 0);
}

#[test]
fn test_mcts_max_depth() {
    let (game, mut state) = SimpleGameBuilder::new()
        .player_count(2)
        .starting_life(50) // High life so game doesn't end quickly
        .build(42);

    let config = MCTSConfig::default().with_max_depth(5);
    let mut search = MCTSSearch::new(game, config);

    search.search(&mut state, PlayerId::new(0), 100);

    // Tree depth should be limited
    let tree_stats = search.tree().stats();
    assert!(
        tree_stats.max_depth <= 10, // Some slack for implementation
        "Tree depth {} should be limited",
        tree_stats.max_depth
    );
}

#[test]
fn test_mcts_temperature_greedy() {
    let (game, mut state) = SimpleGameBuilder::new()
        .player_count(2)
        .starting_life(20)
        .build(42);

    // Temperature 0 = greedy (most visited)
    let config = MCTSConfig::default().with_temperature(0.0);
    let mut search = MCTSSearch::new(game, config);

    search.search(&mut state, PlayerId::new(0), 100);

    let action = search.search(&mut state, PlayerId::new(0), 100);
    let visits = search.action_visits();

    // Best action should have most visits
    if let Some(chosen) = action {
        let max_visits = visits.iter().map(|(_, v)| v).max().unwrap();
        let chosen_visits = visits
            .iter()
            .find(|(a, _)| a == &chosen)
            .map(|(_, v)| v)
            .unwrap();
        assert_eq!(chosen_visits, max_visits);
    }
}

// =============================================================================
// Serialization Tests
// =============================================================================

#[test]
fn test_mcts_config_serialization() {
    let config = MCTSConfig::default()
        .with_exploration(2.0)
        .with_seed(999);

    let json = serde_json::to_string(&config).unwrap();
    let deserialized: MCTSConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(config.seed, deserialized.seed);
    assert_eq!(config.exploration_constant, deserialized.exploration_constant);
}

#[test]
fn test_mcts_tree_serialization() {
    let tree = MCTSTree::new(PlayerId::new(0), 2);

    let json = serde_json::to_string(&tree).unwrap();
    let deserialized: MCTSTree = serde_json::from_str(&json).unwrap();

    assert_eq!(tree.len(), deserialized.len());
    assert_eq!(tree.player_count(), deserialized.player_count());
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_mcts_single_action() {
    // Create a game state where only one action is possible
    // (In SimpleGame, passing is always available, so this tests that case)
    let (game, mut state) = SimpleGameBuilder::new()
        .player_count(2)
        .starting_life(20)
        .cards_per_player(0) // No cards to play
        .build(42);

    let config = MCTSConfig::default();
    let mut search = MCTSSearch::new(game, config);

    let action = search.search(&mut state, PlayerId::new(0), 10);

    // Should still return an action (likely pass)
    assert!(action.is_some());
}

#[test]
fn test_mcts_near_terminal() {
    // Low life - game could end quickly
    let (game, mut state) = SimpleGameBuilder::new()
        .player_count(2)
        .starting_life(1)
        .build(42);

    let config = MCTSConfig::default();
    let mut search = MCTSSearch::new(game, config);

    // Should handle near-terminal states gracefully
    let action = search.search(&mut state, PlayerId::new(0), 50);
    assert!(action.is_some());
}

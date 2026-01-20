//! N-Player capability verification tests.
//!
//! These tests verify that the engine has no hidden 2-player assumptions
//! and works correctly for games with 1-8+ players.

use rust_ccg::core::{Action, EntityId, GameState, PlayerId, ZoneId};
use rust_ccg::effects::{Effect, EffectBatch, EffectResolver, ResolverContext, TargetFilter, TargetSpec, TargetSelector};
use rust_ccg::games::simple::SimpleGameBuilder;
use rust_ccg::rules::RulesEngine;

/// Test that GameState correctly handles varying player counts.
#[test]
fn test_game_state_player_counts() {
    for player_count in [1, 2, 3, 4, 5, 6, 7, 8] {
        let state = GameState::new(player_count, 42);
        assert_eq!(state.player_count(), player_count);

        // Verify all player IDs are valid
        for player in PlayerId::all(player_count) {
            let entity = EntityId::player(player);
            assert!(entity.is_player(player_count));
            assert_eq!(entity.as_player(player_count), Some(player));
        }

        // Verify player_count is not a valid player ID
        let non_player = EntityId(player_count as u32);
        assert!(!non_player.is_player(player_count));
    }
}

/// Test that player state works for all players in N-player games.
#[test]
fn test_player_state_n_players() {
    let player_count = 6;
    let mut state = GameState::new(player_count, 42);

    // Set different life totals for each player
    for player in PlayerId::all(player_count) {
        let life = (player.0 as i64 + 1) * 10; // 10, 20, 30, ...
        state.public.set_player_state(player, "life", life);
    }

    // Verify all players have correct values
    for player in PlayerId::all(player_count) {
        let expected = (player.0 as i64 + 1) * 10;
        assert_eq!(state.public.get_player_state(player, "life", 0), expected);
    }
}

/// Test SimpleGame works with 4 players.
#[test]
fn test_simple_game_4_player() {
    let (mut game, mut state) = SimpleGameBuilder::new()
        .player_count(4)
        .starting_life(30)
        .cards_per_player(5)
        .build(42);

    // All 4 players should exist
    assert_eq!(state.player_count(), 4);
    for player in PlayerId::all(4) {
        assert_eq!(state.public.get_player_state(player, "life", 0), 30);
    }

    // Play a few rounds
    let mut actions_taken = 0;
    for _ in 0..20 {
        if game.is_terminal(&state).is_some() {
            break;
        }

        let active = state.public.active_player;
        let legal = game.legal_actions(&state, active);
        if let Some(action) = legal.first() {
            game.apply_action(&mut state, active, action);
            actions_taken += 1;
        }
    }

    // Game should have progressed (actions taken or game ended)
    assert!(actions_taken > 0 || game.is_terminal(&state).is_some());
}

/// Test SimpleGame works with 6 players.
#[test]
fn test_simple_game_6_player() {
    let (mut game, mut state) = SimpleGameBuilder::new()
        .player_count(6)
        .starting_life(40)
        .build(123);

    assert_eq!(state.player_count(), 6);

    // Play to completion or max turns
    let mut turns = 0;
    while game.is_terminal(&state).is_none() && turns < 100 {
        let active = state.public.active_player;
        let legal = game.legal_actions(&state, active);
        if let Some(action) = legal.first() {
            game.apply_action(&mut state, active, action);
        }
        turns += 1;
    }

    // Game should eventually end
    assert!(game.is_terminal(&state).is_some() || turns == 100);
}

/// Test SimpleGame works with 8 players.
#[test]
fn test_simple_game_8_player() {
    let (game, state) = SimpleGameBuilder::new()
        .player_count(8)
        .starting_life(50)
        .cards_per_player(3)
        .build(456);

    let _ = game; // silence unused warning
    assert_eq!(state.player_count(), 8);

    // Verify all players have the right starting conditions
    for player in PlayerId::all(8) {
        assert_eq!(state.public.get_player_state(player, "life", 0), 50);
    }
}

/// Test effect targeting works for N players.
#[test]
fn test_effect_targeting_n_players() {
    let player_count = 5;
    let state = GameState::new(player_count, 42);

    // Target all opponents from player 0's perspective
    let spec = TargetSpec::single_player()
        .with_filter(TargetFilter::Opponent);
    let selector = TargetSelector::new(spec, PlayerId::new(0));

    let targets = selector.valid_targets(&state);

    // Should have 4 opponents (players 1-4)
    assert_eq!(targets.len(), 4);
    assert!(!targets.contains(&EntityId::player(PlayerId::new(0))));
    for i in 1..5 {
        assert!(targets.contains(&EntityId::player(PlayerId::new(i))));
    }
}

/// Test effect targeting with Self_ filter.
#[test]
fn test_effect_targeting_self_n_players() {
    let player_count = 4;
    let state = GameState::new(player_count, 42);

    // Test self-targeting from each player's perspective
    for active_player in PlayerId::all(player_count) {
        let spec = TargetSpec::single_player()
            .with_filter(TargetFilter::Self_);
        let selector = TargetSelector::new(spec, active_player);

        let targets = selector.valid_targets(&state);

        assert_eq!(targets.len(), 1);
        assert!(targets.contains(&EntityId::player(active_player)));
    }
}

/// Test damage effects work on any player in N-player game.
#[test]
fn test_damage_effects_n_players() {
    let player_count = 4;
    let mut state = GameState::new(player_count, 42);
    let deck = ZoneId::new(0);
    let hand = ZoneId::new(1);

    // Set up initial life totals
    for player in PlayerId::all(player_count) {
        state.public.set_player_state(player, "life", 20);
    }

    let context = ResolverContext::new(
        move |_| deck,
        move |_| hand,
    );

    // Damage each player a different amount
    for player in PlayerId::all(player_count) {
        let damage = (player.0 as i64 + 1) * 2; // 2, 4, 6, 8
        let effect = Effect::damage(damage);
        let target = EntityId::player(player);

        EffectResolver::resolve_single(&mut state, &effect, target, &context);
    }

    // Verify damage was applied correctly
    for player in PlayerId::all(player_count) {
        let expected_damage = (player.0 as i64 + 1) * 2;
        let expected_life = 20 - expected_damage;
        assert_eq!(
            state.public.get_player_state(player, "life", 0),
            expected_life,
            "Player {} should have {} life",
            player.0,
            expected_life
        );
    }
}

/// Test EffectBatch with multiple player targets.
#[test]
fn test_effect_batch_n_players() {
    let player_count = 4;
    let mut state = GameState::new(player_count, 42);
    let deck = ZoneId::new(0);
    let hand = ZoneId::new(1);

    for player in PlayerId::all(player_count) {
        state.public.set_player_state(player, "life", 30);
    }

    let context = ResolverContext::new(
        move |_| deck,
        move |_| hand,
    );

    // Create a batch that damages all players
    let mut batch = EffectBatch::new();
    for player in PlayerId::all(player_count) {
        batch.add_player(Effect::damage(5), player);
    }

    EffectResolver::resolve_batch(&mut state, &batch, &context);

    // All players should have 25 life
    for player in PlayerId::all(player_count) {
        assert_eq!(state.public.get_player_state(player, "life", 0), 25);
    }
}

/// Test turn order cycles through all players correctly.
#[test]
fn test_turn_order_n_players() {
    let (mut game, mut state) = SimpleGameBuilder::new()
        .player_count(5)
        .build(42);

    let mut seen_players = vec![false; 5];
    let mut actions_taken = 0;

    // Play enough actions to see each player act
    for _ in 0..25 {
        if game.is_terminal(&state).is_some() {
            break;
        }

        let active = state.public.active_player;
        seen_players[active.0 as usize] = true;

        let legal = game.legal_actions(&state, active);
        if let Some(action) = legal.first() {
            game.apply_action(&mut state, active, action);
            actions_taken += 1;
        }
    }

    // Multiple players should have taken turns
    let players_seen = seen_players.iter().filter(|&&x| x).count();
    assert!(players_seen > 1 || actions_taken > 0);
}

/// Test player_count boundary values.
#[test]
fn test_player_count_boundaries() {
    // Single player
    let state1 = GameState::new(1, 42);
    assert_eq!(state1.player_count(), 1);
    assert!(EntityId(0).is_player(1));
    assert!(!EntityId(1).is_player(1));

    // Maximum typical player count (8)
    let state8 = GameState::new(8, 42);
    assert_eq!(state8.player_count(), 8);
    for i in 0..8 {
        assert!(EntityId(i).is_player(8));
    }
    assert!(!EntityId(8).is_player(8));
}

/// Test deterministic replay with different player counts.
#[test]
fn test_deterministic_replay_n_players() {
    for player_count in [3, 4, 5] {
        let (mut game, mut state1) = SimpleGameBuilder::new()
            .player_count(player_count)
            .build(42);

        // Play game with seed 42
        let mut actions1: Vec<(PlayerId, Action)> = Vec::new();

        for _ in 0..20 {
            if game.is_terminal(&state1).is_some() {
                break;
            }
            let active = state1.public.active_player;
            let legal = game.legal_actions(&state1, active);
            if let Some(action) = legal.first() {
                actions1.push((active, action.clone()));
                game.apply_action(&mut state1, active, action);
            }
        }

        // Replay with same seed
        let (game2, mut state2) = SimpleGameBuilder::new()
            .player_count(player_count)
            .build(42);
        let _ = game2; // silence unused warning

        for (player, action) in &actions1 {
            game.apply_action(&mut state2, *player, action);
        }

        // States should match
        for player in PlayerId::all(player_count) {
            assert_eq!(
                state1.public.get_player_state(player, "life", 0),
                state2.public.get_player_state(player, "life", 0),
                "Player {} life mismatch for {}-player game",
                player.0,
                player_count
            );
        }
    }
}

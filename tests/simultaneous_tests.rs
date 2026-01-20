//! Simultaneous action tests.
//!
//! These tests verify support for games where multiple players
//! act simultaneously (like Sushi Go or drafting phases).

use rust_ccg::cards::{CardId, CardInstance};
use rust_ccg::core::{GameState, PlayerId, ZoneId};

/// Test setting multiple players as having priority.
#[test]
fn test_multiple_priority_players() {
    let player_count = 4;
    let mut state = GameState::new(player_count, 42);

    // In a simultaneous game, all players have priority at once
    let all_players: Vec<PlayerId> = PlayerId::all(player_count).collect();
    state.public.priority_players = all_players.clone();

    assert_eq!(state.public.priority_players.len(), 4);
    for player in PlayerId::all(player_count) {
        assert!(state.public.priority_players.contains(&player));
    }
}

/// Test clearing and resetting priority for round-based simultaneous play.
#[test]
fn test_priority_round_management() {
    let player_count = 3;
    let mut state = GameState::new(player_count, 42);

    // Start of round: all players have priority
    let all_players: Vec<PlayerId> = PlayerId::all(player_count).collect();
    state.public.priority_players = all_players.clone();
    assert_eq!(state.public.priority_players.len(), 3);

    // Player 0 makes their choice (remove from priority)
    state.public.priority_players.retain(|&p| p != PlayerId::new(0));
    assert_eq!(state.public.priority_players.len(), 2);
    assert!(!state.public.priority_players.contains(&PlayerId::new(0)));

    // Player 2 makes their choice
    state.public.priority_players.retain(|&p| p != PlayerId::new(2));
    assert_eq!(state.public.priority_players.len(), 1);

    // Player 1 makes their choice
    state.public.priority_players.retain(|&p| p != PlayerId::new(1));
    assert!(state.public.priority_players.is_empty());

    // All players have acted - next round starts, restore priority
    state.public.priority_players = all_players;
    assert_eq!(state.public.priority_players.len(), 3);
}

/// Test simultaneous card drafting scenario.
#[test]
fn test_simultaneous_draft() {
    let player_count = 4;
    let mut state = GameState::new(player_count, 42);

    // Create draft pack zones for each player
    let pack_zones: Vec<ZoneId> = (0..player_count)
        .map(|i| ZoneId::new(i as u16))
        .collect();

    // Create hand zones for each player
    let hand_zones: Vec<ZoneId> = (0..player_count)
        .map(|i| ZoneId::new((player_count + i) as u16))
        .collect();

    // Initialize ordered zones for packs
    for &pack in &pack_zones {
        state.zones.init_ordered_zone(pack);
    }

    // Create cards in each pack (simulating draft start)
    for (player_idx, &pack) in pack_zones.iter().enumerate() {
        for card_idx in 0..5 {
            let entity_id = state.alloc_entity();
            let card = CardInstance::new(
                entity_id,
                CardId::new((player_idx * 5 + card_idx) as u32 + 1),
                PlayerId::new(player_idx as u8),
                pack,
            );
            state.add_card(card);
        }
    }

    // Verify pack sizes
    for &pack in &pack_zones {
        assert_eq!(state.zones.zone_size(pack), 5);
    }

    // All players have priority (simultaneous selection)
    state.public.priority_players = PlayerId::all(player_count).collect();

    // Simulate each player selecting a card simultaneously
    // (In a real game, this would be handled by collecting all choices first)
    for (&pack, &hand) in pack_zones.iter().zip(hand_zones.iter()) {
        // Each player takes the top card from their current pack
        if let Some(card_entity) = state.zones.pop_top(pack) {
            state.zones.add_to_zone(card_entity, hand, None);
            if let Some(card) = state.get_card_mut(card_entity) {
                card.zone = hand;
            }
        }
    }

    // Verify each pack lost one card
    for &pack in &pack_zones {
        assert_eq!(state.zones.zone_size(pack), 4);
    }

    // Verify each hand gained one card
    for &hand in &hand_zones {
        assert_eq!(state.zones.zone_size(hand), 1);
    }
}

/// Test tracking submitted actions in simultaneous play.
#[test]
fn test_simultaneous_action_tracking() {
    let player_count = 3;
    let mut state = GameState::new(player_count, 42);

    // Use turn_state to track which players have submitted actions
    // Key: "submitted_P{N}" = 1 if submitted, 0 otherwise
    for player in PlayerId::all(player_count) {
        let key = format!("submitted_p{}", player.0);
        state.public.set_turn_state(&key, 0);
    }

    // All players have priority
    state.public.priority_players = PlayerId::all(player_count).collect();

    // Player 1 submits their action
    state.public.set_turn_state("submitted_p1", 1);
    assert_eq!(state.public.get_turn_state("submitted_p1", 0), 1);
    assert_eq!(state.public.get_turn_state("submitted_p0", 0), 0);
    assert_eq!(state.public.get_turn_state("submitted_p2", 0), 0);

    // Player 0 and 2 submit
    state.public.set_turn_state("submitted_p0", 1);
    state.public.set_turn_state("submitted_p2", 1);

    // Check all submitted
    let all_submitted = PlayerId::all(player_count).all(|p| {
        let key = format!("submitted_p{}", p.0);
        state.public.get_turn_state(&key, 0) == 1
    });
    assert!(all_submitted);
}

/// Test reveal phase (all actions resolve simultaneously).
#[test]
fn test_reveal_phase() {
    let player_count = 2;
    let mut state = GameState::new(player_count, 42);

    // Track chosen cards for each player (using player state)
    state.public.set_player_state(PlayerId::new(0), "chosen_card", 5);
    state.public.set_player_state(PlayerId::new(1), "chosen_card", 3);

    // Simulate reveal: both players' choices become visible
    // In a real game, this might move cards from "hidden selection" to "revealed"

    // Read all choices (they're now public)
    let player0_choice = state.public.get_player_state(PlayerId::new(0), "chosen_card", 0);
    let player1_choice = state.public.get_player_state(PlayerId::new(1), "chosen_card", 0);

    assert_eq!(player0_choice, 5);
    assert_eq!(player1_choice, 3);

    // Apply effects based on revealed choices
    // (Simplified: just track who "won" this round)
    if player0_choice > player1_choice {
        state.public.modify_player_state(PlayerId::new(0), "score", 1);
    } else {
        state.public.modify_player_state(PlayerId::new(1), "score", 1);
    }

    assert_eq!(state.public.get_player_state(PlayerId::new(0), "score", 0), 1);
    assert_eq!(state.public.get_player_state(PlayerId::new(1), "score", 0), 0);
}

/// Test N-player simultaneous action (like Sushi Go).
#[test]
fn test_n_player_simultaneous() {
    let player_count = 5;
    let mut state = GameState::new(player_count, 42);

    // All players pick simultaneously
    state.public.priority_players = PlayerId::all(player_count).collect();

    // Each player makes a hidden selection (stored in player state)
    for player in PlayerId::all(player_count) {
        let choice = (player.0 + 1) as i64; // Player 0 picks 1, player 1 picks 2, etc.
        state.public.set_player_state(player, "hidden_choice", choice);
    }

    // All players have "submitted"
    state.public.priority_players.clear();

    // Reveal phase: process all selections
    let mut total_picks = 0;
    for player in PlayerId::all(player_count) {
        let choice = state.public.get_player_state(player, "hidden_choice", 0);
        total_picks += choice;
    }

    // 1 + 2 + 3 + 4 + 5 = 15
    assert_eq!(total_picks, 15);
}

/// Test turn state clearing between simultaneous rounds.
#[test]
fn test_round_state_management() {
    let player_count = 4;
    let mut state = GameState::new(player_count, 42);

    // Round 1: set up turn state
    state.public.set_turn_state("round", 1);
    for player in PlayerId::all(player_count) {
        let key = format!("submitted_p{}", player.0);
        state.public.set_turn_state(&key, 0);
    }

    // Players submit
    for player in PlayerId::all(player_count) {
        let key = format!("submitted_p{}", player.0);
        state.public.set_turn_state(&key, 1);
    }

    // End of round - advance turn (clears turn state)
    state.public.advance_turn();

    // Turn state should be cleared
    assert_eq!(state.public.get_turn_state("round", 0), 0);
    for player in PlayerId::all(player_count) {
        let key = format!("submitted_p{}", player.0);
        assert_eq!(state.public.get_turn_state(&key, 0), 0);
    }

    // Set up round 2
    state.public.set_turn_state("round", 2);
    assert_eq!(state.public.get_turn_state("round", 0), 2);
}

/// Test simultaneous with turn preservation (some state persists).
#[test]
fn test_round_state_preservation() {
    let player_count = 3;
    let mut state = GameState::new(player_count, 42);

    // Track permanent state in turn_state (like round number)
    state.public.set_turn_state("permanent_round", 1);

    // Advance but preserve state
    state.public.advance_turn_preserve_state();

    // State should persist
    assert_eq!(state.public.get_turn_state("permanent_round", 0), 1);
}

/// Test mixed turn order (some sequential, some simultaneous phases).
#[test]
fn test_mixed_phases() {
    let player_count = 4;
    let mut state = GameState::new(player_count, 42);

    // Phase 1: Simultaneous selection
    state.public.priority_players = PlayerId::all(player_count).collect();
    assert_eq!(state.public.priority_players.len(), 4);

    // All players make selections...
    // (simulate by clearing priority)
    state.public.priority_players.clear();

    // Phase 2: Sequential resolution (player order)
    for player in PlayerId::all(player_count) {
        state.public.priority_players = vec![player];

        // Process this player's action...
        state.public.modify_player_state(player, "processed", 1);

        state.public.priority_players.clear();
    }

    // Verify all players were processed
    for player in PlayerId::all(player_count) {
        assert_eq!(state.public.get_player_state(player, "processed", 0), 1);
    }
}

//! Deckbuilder pattern tests.
//!
//! These tests verify support for deckbuilder games like Dominion:
//! - Neutral cards (no owner)
//! - Shared zones (market, supply)
//! - Card acquisition mechanics

use rust_ccg::cards::{CardId, CardInstance};
use rust_ccg::core::{GameState, PlayerId, ZoneId};

/// Test creating neutral (unowned) cards.
#[test]
fn test_neutral_cards() {
    let mut state = GameState::new(2, 42);
    let market_zone = ZoneId::new(0);

    // Create neutral market cards
    for i in 0..5 {
        let entity_id = state.alloc_entity();
        let card = CardInstance::neutral(entity_id, CardId::new(i + 1), market_zone);

        assert!(card.is_neutral());
        assert_eq!(card.owner, None);
        assert_eq!(card.controller, None);

        state.add_card(card);
    }

    // Verify cards are in the zone
    assert_eq!(state.zones.zone_size(market_zone), 5);

    // All cards should be neutral
    for entity in state.zones.cards_in_zone(market_zone) {
        let card = state.get_card(entity).expect("Card should exist");
        assert!(card.is_neutral());
    }
}

/// Test acquiring cards from a shared market zone.
#[test]
fn test_card_acquisition() {
    let mut state = GameState::new(2, 42);
    let market = ZoneId::new(0);
    let player0_deck = ZoneId::new(1);
    let player1_deck = ZoneId::new(2);

    state.zones.init_ordered_zone(player0_deck);
    state.zones.init_ordered_zone(player1_deck);

    // Create 3 neutral cards in the market
    let mut market_cards = Vec::new();
    for i in 0..3 {
        let entity_id = state.alloc_entity();
        let card = CardInstance::neutral(entity_id, CardId::new(i + 1), market);
        market_cards.push(entity_id);
        state.add_card(card);
    }

    // Player 0 acquires card 0
    let acquired_card = market_cards[0];
    state.zones.move_to_zone(acquired_card, player0_deck, None);
    if let Some(card) = state.get_card_mut(acquired_card) {
        card.owner = Some(PlayerId::new(0));
        card.controller = Some(PlayerId::new(0));
        card.zone = player0_deck;
    }

    // Player 1 acquires card 1
    let acquired_card2 = market_cards[1];
    state.zones.move_to_zone(acquired_card2, player1_deck, None);
    if let Some(card) = state.get_card_mut(acquired_card2) {
        card.owner = Some(PlayerId::new(1));
        card.controller = Some(PlayerId::new(1));
        card.zone = player1_deck;
    }

    // Verify zone populations
    assert_eq!(state.zones.zone_size(market), 1); // 1 card left
    assert_eq!(state.zones.zone_size(player0_deck), 1);
    assert_eq!(state.zones.zone_size(player1_deck), 1);

    // Verify ownership changes
    let p0_card = state.get_card(market_cards[0]).unwrap();
    assert_eq!(p0_card.owner, Some(PlayerId::new(0)));
    assert!(!p0_card.is_neutral());

    let p1_card = state.get_card(market_cards[1]).unwrap();
    assert_eq!(p1_card.owner, Some(PlayerId::new(1)));
    assert!(!p1_card.is_neutral());

    // Market card remains neutral
    let market_card = state.get_card(market_cards[2]).unwrap();
    assert!(market_card.is_neutral());
}

/// Test shared zone with multiple access points.
#[test]
fn test_shared_zone_access() {
    let player_count = 4;
    let mut state = GameState::new(player_count, 42);
    let shared_supply = ZoneId::new(0);

    // Create supply cards
    let mut supply_entities = Vec::new();
    for i in 0..10 {
        let entity_id = state.alloc_entity();
        let card = CardInstance::neutral(entity_id, CardId::new(i + 1), shared_supply);
        supply_entities.push(entity_id);
        state.add_card(card);
    }

    // Each player takes from the same supply
    let player_discard_zones: Vec<ZoneId> = (0..player_count)
        .map(|i| ZoneId::new((i + 1) as u16))
        .collect();

    for (i, &discard_zone) in player_discard_zones.iter().enumerate() {
        state.zones.init_ordered_zone(discard_zone);

        // Player takes a card from supply
        let taken_card = supply_entities[i];
        state.zones.move_to_zone(taken_card, discard_zone, None);
        if let Some(card) = state.get_card_mut(taken_card) {
            card.owner = Some(PlayerId::new(i as u8));
            card.zone = discard_zone;
        }
    }

    // Supply should have 6 cards left (10 - 4 players)
    assert_eq!(state.zones.zone_size(shared_supply), 6);

    // Each player should have 1 card in discard
    for discard_zone in player_discard_zones {
        assert_eq!(state.zones.zone_size(discard_zone), 1);
    }
}

/// Test controller vs owner distinction.
#[test]
fn test_controller_vs_owner() {
    let mut state = GameState::new(2, 42);
    let battlefield = ZoneId::new(0);

    let entity_id = state.alloc_entity();
    let mut card = CardInstance::new(entity_id, CardId::new(1), PlayerId::new(0), battlefield);

    // Initially owner and controller are the same
    assert_eq!(card.owner, Some(PlayerId::new(0)));
    assert_eq!(card.controller, Some(PlayerId::new(0)));

    // Simulate control-changing effect
    card.controller = Some(PlayerId::new(1));

    // Owner unchanged, controller changed
    assert_eq!(card.owner, Some(PlayerId::new(0)));
    assert_eq!(card.controller, Some(PlayerId::new(1)));

    state.add_card(card);

    // Verify from state
    let retrieved = state.get_card(entity_id).unwrap();
    assert_eq!(retrieved.owner, Some(PlayerId::new(0)));
    assert_eq!(retrieved.controller, Some(PlayerId::new(1)));
}

/// Test card shuffling into deck (deckbuilder mechanic).
#[test]
fn test_shuffle_into_deck() {
    let mut state = GameState::new(2, 42);
    let discard = ZoneId::new(0);
    let deck = ZoneId::new(1);

    state.zones.init_ordered_zone(discard);
    state.zones.init_ordered_zone(deck);

    // Add cards to discard pile
    let mut discard_cards = Vec::new();
    for i in 0..5 {
        let entity_id = state.alloc_entity();
        let card = CardInstance::new(entity_id, CardId::new(i + 1), PlayerId::new(0), discard);
        discard_cards.push(entity_id);
        state.add_card(card);
    }

    // Move all cards from discard to deck
    for &card_entity in &discard_cards {
        state.zones.move_to_zone(card_entity, deck, None);
        if let Some(card) = state.get_card_mut(card_entity) {
            card.zone = deck;
        }
    }

    assert_eq!(state.zones.zone_size(discard), 0);
    assert_eq!(state.zones.zone_size(deck), 5);

    // Shuffle the deck
    state.zones.shuffle_zone(deck, &mut state.rng);

    // Still has 5 cards
    assert_eq!(state.zones.zone_size(deck), 5);
}

/// Test neutral card with face-down state (for hidden supply piles).
#[test]
fn test_face_down_supply() {
    let mut state = GameState::new(2, 42);
    let supply = ZoneId::new(0);

    let entity_id = state.alloc_entity();
    let mut card = CardInstance::neutral(entity_id, CardId::new(1), supply);

    // Set face down (hidden in supply pile)
    card.face_down = true;

    state.add_card(card);

    let retrieved = state.get_card(entity_id).unwrap();
    assert!(retrieved.face_down);
    assert!(retrieved.is_neutral());
}

/// Test N-player deckbuilder scenario.
#[test]
fn test_n_player_deckbuilder() {
    let player_count = 4;
    let mut state = GameState::new(player_count, 42);

    // Zone IDs: 0 = shared supply, 1-4 = player decks, 5-8 = player discards, 9-12 = player hands
    let supply = ZoneId::new(0);

    // Initialize player zones
    let player_zones: Vec<(ZoneId, ZoneId, ZoneId)> = (0..player_count)
        .map(|i| {
            let deck = ZoneId::new((1 + i) as u16);
            let discard = ZoneId::new((5 + i) as u16);
            let hand = ZoneId::new((9 + i) as u16);
            state.zones.init_ordered_zone(deck);
            state.zones.init_ordered_zone(discard);
            (deck, discard, hand)
        })
        .collect();

    // Create supply cards (neutral)
    for i in 0..20 {
        let entity_id = state.alloc_entity();
        let card = CardInstance::neutral(entity_id, CardId::new((i % 5) + 1), supply);
        state.add_card(card);
    }

    // Create starting deck cards for each player
    for (i, &(deck, _, _)) in player_zones.iter().enumerate() {
        let player = PlayerId::new(i as u8);
        for j in 0..7 {
            let entity_id = state.alloc_entity();
            let card = CardInstance::new(entity_id, CardId::new(100 + j), player, deck);
            state.add_card(card);
        }
    }

    // Verify setup
    assert_eq!(state.zones.zone_size(supply), 20);
    for (deck, _, _) in &player_zones {
        assert_eq!(state.zones.zone_size(*deck), 7);
    }
}

/// Test card state tracking for deckbuilder mechanics (buy cost tracking, etc).
#[test]
fn test_card_state_tracking() {
    let mut state = GameState::new(2, 42);
    let supply = ZoneId::new(0);

    // Create a card with custom state for tracking
    let entity_id = state.alloc_entity();
    let mut card = CardInstance::neutral(entity_id, CardId::new(1), supply);

    // Track supply count (how many copies left)
    card.set_state("supply_count", 10);

    state.add_card(card);

    // Simulate buying one copy
    if let Some(c) = state.get_card_mut(entity_id) {
        let count = c.get_state("supply_count", 0);
        c.set_state("supply_count", count - 1);
    }

    let retrieved = state.get_card(entity_id).unwrap();
    assert_eq!(retrieved.get_state("supply_count", 0), 9);
}

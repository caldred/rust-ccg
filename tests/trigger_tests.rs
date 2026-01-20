//! Trigger system integration tests.
//!
//! These tests verify the trigger system works correctly for N players
//! and integrates properly with the effects system.

use rust_ccg::cards::{CardId, CardInstance};
use rust_ccg::core::{EntityId, GameState, PlayerId, ZoneId};
use rust_ccg::effects::Effect;
use rust_ccg::triggers::{
    EventTypeId, GameEvent, Trigger, TriggerCondition,
    TriggerId, TriggerRegistry, TriggerTiming,
};

// Define test event types
const DAMAGE_DEALT: EventTypeId = EventTypeId::new(1);
const CARD_PLAYED: EventTypeId = EventTypeId::new(2);
const TURN_START: EventTypeId = EventTypeId::new(3);
const CARD_DRAWN: EventTypeId = EventTypeId::new(4);
const CREATURE_DIED: EventTypeId = EventTypeId::new(5);

/// Test basic trigger firing for N players.
#[test]
fn test_n_player_triggers() {
    let player_count = 4;
    let state = GameState::new(player_count, 42);
    let mut registry = TriggerRegistry::new();

    // Register a "turn start" trigger for each player
    for player_idx in 0..player_count as u8 {
        let trigger = Trigger::new(
            TriggerId::new(player_idx as u32),
            format!("Player{} Turn Start", player_idx),
            TURN_START,
        )
        .with_controller(PlayerId::new(player_idx))
        .with_condition(TriggerCondition::ForPlayer(PlayerId::new(player_idx)))
        .with_effect(Effect::heal(1));

        registry.register_with_id(trigger);
    }

    // Each player's turn start should only trigger their ability
    for player_idx in 0..player_count as u8 {
        let event = GameEvent::for_player(TURN_START, PlayerId::new(player_idx));
        let triggered = registry.find_triggers(&event, &state, TriggerTiming::After, None);

        assert_eq!(
            triggered.len(),
            1,
            "Player {} should have exactly 1 trigger fire",
            player_idx
        );
        assert_eq!(
            triggered[0].controller,
            Some(PlayerId::new(player_idx)),
            "Triggered effect should be controlled by player {}",
            player_idx
        );
    }
}

/// Test triggers that affect all players.
#[test]
fn test_global_trigger() {
    let player_count = 3;
    let state = GameState::new(player_count, 42);
    let mut registry = TriggerRegistry::new();

    // A global trigger that fires for any player's turn start
    let trigger = Trigger::new(TriggerId::new(1), "Global Turn Start", TURN_START)
        .with_condition(TriggerCondition::Always)
        .with_effect(Effect::modify_player("turns_taken", 1));

    registry.register_with_id(trigger);

    // Should trigger for any player
    for player_idx in 0..player_count as u8 {
        let event = GameEvent::for_player(TURN_START, PlayerId::new(player_idx));
        let triggered = registry.find_triggers(&event, &state, TriggerTiming::After, None);

        assert_eq!(triggered.len(), 1, "Global trigger should fire for player {}", player_idx);
    }
}

/// Test damage-based triggers.
#[test]
fn test_damage_trigger() {
    let mut state = GameState::new(2, 42);
    let mut registry = TriggerRegistry::new();
    let battlefield = ZoneId::new(0);

    // Create a creature with lifelink (heals when dealing damage)
    let creature = state.alloc_entity();
    let card = CardInstance::new(creature, CardId::new(1), PlayerId::new(0), battlefield);
    state.add_card(card);

    // Register lifelink trigger
    let lifelink = Trigger::new(TriggerId::new(1), "Lifelink", DAMAGE_DEALT)
        .with_source(creature)
        .with_controller(PlayerId::new(0))
        .with_condition(TriggerCondition::SourceIs(creature))
        .with_effect(Effect::heal(1)); // Heal amount would be derived from damage in real game

    registry.register_with_id(lifelink);

    // When creature deals damage
    let event = GameEvent::damage(DAMAGE_DEALT, creature, EntityId::player_id(1), 3);
    let triggered = registry.find_triggers(&event, &state, TriggerTiming::After, None);

    assert_eq!(triggered.len(), 1);
    assert_eq!(triggered[0].source, Some(creature));
}

/// Test trigger with value conditions.
#[test]
fn test_value_condition_trigger() {
    let state = GameState::new(2, 42);
    let mut registry = TriggerRegistry::new();

    // Trigger only on "big" damage (>= 5)
    let big_damage = Trigger::new(TriggerId::new(1), "Big Damage Response", DAMAGE_DEALT)
        .with_condition(TriggerCondition::ValueAtLeast { index: 0, min: 5 })
        .with_effect(Effect::draw(1));

    registry.register_with_id(big_damage);

    // Small damage shouldn't trigger
    let small = GameEvent::new(DAMAGE_DEALT).with_value(3);
    let triggered = registry.find_triggers(&small, &state, TriggerTiming::After, None);
    assert_eq!(triggered.len(), 0);

    // Big damage should trigger
    let big = GameEvent::new(DAMAGE_DEALT).with_value(7);
    let triggered = registry.find_triggers(&big, &state, TriggerTiming::After, None);
    assert_eq!(triggered.len(), 1);
}

/// Test trigger timing (before vs after).
#[test]
fn test_trigger_timing() {
    let state = GameState::new(2, 42);
    let mut registry = TriggerRegistry::new();

    // "Before" trigger (could modify/cancel)
    let before = Trigger::new(TriggerId::new(1), "Shield", DAMAGE_DEALT)
        .with_timing(TriggerTiming::Before)
        .with_effect(Effect::modify_player("shield", 1));

    // "After" trigger (react to damage)
    let after = Trigger::new(TriggerId::new(2), "Vengeance", DAMAGE_DEALT)
        .with_timing(TriggerTiming::After)
        .with_effect(Effect::damage(2));

    registry.register_with_id(before);
    registry.register_with_id(after);

    let event = GameEvent::new(DAMAGE_DEALT);

    let before_triggers = registry.find_triggers(&event, &state, TriggerTiming::Before, None);
    assert_eq!(before_triggers.len(), 1);
    assert_eq!(before_triggers[0].trigger_id, TriggerId::new(1));

    let after_triggers = registry.find_triggers(&event, &state, TriggerTiming::After, None);
    assert_eq!(after_triggers.len(), 1);
    assert_eq!(after_triggers[0].trigger_id, TriggerId::new(2));
}

/// Test trigger with multiple event types.
#[test]
fn test_multi_event_trigger() {
    let state = GameState::new(2, 42);
    let mut registry = TriggerRegistry::new();

    // Trigger on card played OR card drawn
    let trigger = Trigger::new(TriggerId::new(1), "Card Activity", CARD_PLAYED)
        .also_on(CARD_DRAWN)
        .with_effect(Effect::modify_player("cards_seen", 1));

    registry.register_with_id(trigger);

    // Should fire on card played
    let play_event = GameEvent::new(CARD_PLAYED);
    let triggered = registry.find_triggers(&play_event, &state, TriggerTiming::After, None);
    assert_eq!(triggered.len(), 1);

    // Should fire on card drawn
    let draw_event = GameEvent::new(CARD_DRAWN);
    let triggered = registry.find_triggers(&draw_event, &state, TriggerTiming::After, None);
    assert_eq!(triggered.len(), 1);

    // Should NOT fire on other events
    let other_event = GameEvent::new(CREATURE_DIED);
    let triggered = registry.find_triggers(&other_event, &state, TriggerTiming::After, None);
    assert_eq!(triggered.len(), 0);
}

/// Test limited-use triggers.
#[test]
fn test_limited_use_trigger() {
    let state = GameState::new(2, 42);
    let mut registry = TriggerRegistry::new();

    // Trigger can only fire twice
    let limited = Trigger::new(TriggerId::new(1), "Limited Shield", DAMAGE_DEALT)
        .with_uses(2)
        .with_effect(Effect::heal(1));

    registry.register_with_id(limited);

    let event = GameEvent::new(DAMAGE_DEALT);

    // First use
    let triggered = registry.find_triggers(&event, &state, TriggerTiming::After, None);
    assert_eq!(triggered.len(), 1);
    registry.get_mut(TriggerId::new(1)).unwrap().use_trigger();

    // Second use
    let triggered = registry.find_triggers(&event, &state, TriggerTiming::After, None);
    assert_eq!(triggered.len(), 1);
    registry.get_mut(TriggerId::new(1)).unwrap().use_trigger();

    // Third attempt - should be exhausted
    let triggered = registry.find_triggers(&event, &state, TriggerTiming::After, None);
    assert_eq!(triggered.len(), 0);
}

/// Test removing triggers when source leaves play.
#[test]
fn test_remove_triggers_on_death() {
    let mut state = GameState::new(2, 42);
    let mut registry = TriggerRegistry::new();
    let battlefield = ZoneId::new(0);

    // Create two creatures
    let creature1 = state.alloc_entity();
    let creature2 = state.alloc_entity();
    state.add_card(CardInstance::new(creature1, CardId::new(1), PlayerId::new(0), battlefield));
    state.add_card(CardInstance::new(creature2, CardId::new(2), PlayerId::new(1), battlefield));

    // Each creature has a trigger
    let t1 = Trigger::new(TriggerId::new(1), "Creature1 Ability", DAMAGE_DEALT)
        .with_source(creature1);
    let t2 = Trigger::new(TriggerId::new(2), "Creature2 Ability", DAMAGE_DEALT)
        .with_source(creature2);

    registry.register_with_id(t1);
    registry.register_with_id(t2);

    assert_eq!(registry.len(), 2);

    // Creature 1 dies - remove its triggers
    registry.remove_for_source(creature1);

    assert_eq!(registry.len(), 1);
    assert!(registry.get(TriggerId::new(1)).is_none());
    assert!(registry.get(TriggerId::new(2)).is_some());
}

/// Test complex condition combinations.
#[test]
fn test_complex_conditions() {
    let state = GameState::new(4, 42);
    let mut registry = TriggerRegistry::new();

    // Trigger with complex condition:
    // Event must be DAMAGE_DEALT AND (for player 0 OR for player 1) AND value >= 3
    let condition = TriggerCondition::all([
        TriggerCondition::event(DAMAGE_DEALT),
        TriggerCondition::any([
            TriggerCondition::for_player(PlayerId::new(0)),
            TriggerCondition::for_player(PlayerId::new(1)),
        ]),
        TriggerCondition::value_at_least(0, 3),
    ]);

    let trigger = Trigger::new(TriggerId::new(1), "Complex", DAMAGE_DEALT)
        .with_condition(condition);

    registry.register_with_id(trigger);

    // Should trigger: player 0, damage 5
    let event1 = GameEvent::new(DAMAGE_DEALT)
        .with_player(PlayerId::new(0))
        .with_value(5);
    assert_eq!(
        registry.find_triggers(&event1, &state, TriggerTiming::After, None).len(),
        1
    );

    // Should trigger: player 1, damage 3
    let event2 = GameEvent::new(DAMAGE_DEALT)
        .with_player(PlayerId::new(1))
        .with_value(3);
    assert_eq!(
        registry.find_triggers(&event2, &state, TriggerTiming::After, None).len(),
        1
    );

    // Should NOT trigger: player 2 (wrong player)
    let event3 = GameEvent::new(DAMAGE_DEALT)
        .with_player(PlayerId::new(2))
        .with_value(5);
    assert_eq!(
        registry.find_triggers(&event3, &state, TriggerTiming::After, None).len(),
        0
    );

    // Should NOT trigger: player 0, damage 2 (too low)
    let event4 = GameEvent::new(DAMAGE_DEALT)
        .with_player(PlayerId::new(0))
        .with_value(2);
    assert_eq!(
        registry.find_triggers(&event4, &state, TriggerTiming::After, None).len(),
        0
    );
}

/// Test zone change triggers.
#[test]
fn test_zone_change_triggers() {
    let state = GameState::new(2, 42);
    let mut registry = TriggerRegistry::new();

    const ZONE_CHANGE: EventTypeId = EventTypeId::new(10);
    let hand = ZoneId::new(0);
    let battlefield = ZoneId::new(1);
    let _graveyard = ZoneId::new(2);

    // Trigger when card enters battlefield (zone[1] == battlefield)
    let enters_play = Trigger::new(TriggerId::new(1), "Enters Play", ZONE_CHANGE)
        .with_condition(TriggerCondition::All(vec![
            TriggerCondition::EventType(ZONE_CHANGE),
        ]))
        .with_effect(Effect::heal(1));

    registry.register_with_id(enters_play);

    // Card moves hand -> battlefield
    let event = GameEvent::zone_change(ZONE_CHANGE, EntityId(10), hand, battlefield);
    let triggered = registry.find_triggers(&event, &state, TriggerTiming::After, None);
    assert_eq!(triggered.len(), 1);

    // Verify event has correct zones
    assert_eq!(event.zone(0), Some(hand));
    assert_eq!(event.zone(1), Some(battlefield));
}

/// Test custom condition evaluation.
#[test]
fn test_custom_conditions() {
    let mut state = GameState::new(2, 42);
    state.public.set_player_state(PlayerId::new(0), "life", 5);

    let mut registry = TriggerRegistry::new();

    // Trigger with custom condition "low_life" (life <= 5)
    let trigger = Trigger::new(TriggerId::new(1), "Desperation", DAMAGE_DEALT)
        .with_condition(TriggerCondition::Custom("low_life".to_string()));

    registry.register_with_id(trigger);

    let event = GameEvent::new(DAMAGE_DEALT).with_player(PlayerId::new(0));

    // Without evaluator, custom conditions fail
    let triggered = registry.find_triggers(&event, &state, TriggerTiming::After, None);
    assert_eq!(triggered.len(), 0);

    // With custom evaluator
    let eval = |key: &str, event: &GameEvent, state: &GameState| -> bool {
        if key == "low_life" {
            if let Some(player) = event.player {
                return state.public.get_player_state(player, "life", 20) <= 5;
            }
        }
        false
    };

    let triggered = registry.find_triggers(&event, &state, TriggerTiming::After, Some(&eval));
    assert_eq!(triggered.len(), 1);
}

/// Test trigger enable/disable.
#[test]
fn test_trigger_enable_disable() {
    let state = GameState::new(2, 42);
    let mut registry = TriggerRegistry::new();

    let trigger = Trigger::new(TriggerId::new(1), "Toggleable", DAMAGE_DEALT);
    let id = registry.register_with_id(trigger);

    let event = GameEvent::new(DAMAGE_DEALT);

    // Enabled by default
    assert_eq!(
        registry.find_triggers(&event, &state, TriggerTiming::After, None).len(),
        1
    );

    // Disable
    registry.set_enabled(id, false);
    assert_eq!(
        registry.find_triggers(&event, &state, TriggerTiming::After, None).len(),
        0
    );

    // Re-enable
    registry.set_enabled(id, true);
    assert_eq!(
        registry.find_triggers(&event, &state, TriggerTiming::After, None).len(),
        1
    );
}

/// Test triggers with tags.
#[test]
fn test_tag_based_triggers() {
    let state = GameState::new(2, 42);
    let mut registry = TriggerRegistry::new();

    // Only trigger on combat damage
    let combat_only = Trigger::new(TriggerId::new(1), "Combat Response", DAMAGE_DEALT)
        .with_condition(TriggerCondition::HasTag("combat".to_string()));

    registry.register_with_id(combat_only);

    // Non-combat damage
    let spell_damage = GameEvent::new(DAMAGE_DEALT).with_tag("spell");
    assert_eq!(
        registry.find_triggers(&spell_damage, &state, TriggerTiming::After, None).len(),
        0
    );

    // Combat damage
    let combat_damage = GameEvent::new(DAMAGE_DEALT).with_tag("combat");
    assert_eq!(
        registry.find_triggers(&combat_damage, &state, TriggerTiming::After, None).len(),
        1
    );
}

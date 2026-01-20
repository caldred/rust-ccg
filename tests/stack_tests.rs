//! Stack/Resolution system integration tests.
//!
//! These tests verify both immediate and priority-based resolution
//! systems work correctly with the effect system.

use rust_ccg::core::{GameState, PlayerId, TemplateId, Action};
use rust_ccg::effects::{Effect, EffectBatch, ResolverContext};
use rust_ccg::stack::{
    ImmediateResolution, PriorityStack, ResolutionStatus, ResolutionSystem,
    StackSource,
};
use rust_ccg::triggers::{
    EventTypeId, GameEvent, Trigger, TriggerCondition, TriggerRegistry,
    TriggerTiming, TriggerId,
};

// =============================================================================
// Immediate Resolution Tests
// =============================================================================

/// Test that immediate resolution resolves effects right away.
#[test]
fn test_immediate_resolves_immediately() {
    let mut resolver = ImmediateResolution::new();
    let mut state = GameState::new(2, 42);
    state.public.set_player_state(PlayerId::new(0), "life", 20);
    state.public.set_player_state(PlayerId::new(1), "life", 20);

    // Queue damage to player 1
    let mut batch = EffectBatch::new();
    batch.add_player(Effect::damage(5), PlayerId::new(1));
    resolver.queue_action(Action::new(TemplateId::new(1)), batch, PlayerId::new(0));

    // Process
    let context = ResolverContext::simple(2);
    let status = resolver.process(&mut state, &context);

    assert_eq!(status, ResolutionStatus::Complete);
    assert_eq!(state.public.get_player_state(PlayerId::new(1), "life", 0), 15);
}

/// Test that immediate resolution handles multiple queued effects.
#[test]
fn test_immediate_multiple_effects() {
    let mut resolver = ImmediateResolution::new();
    let mut state = GameState::new(2, 42);
    state.public.set_player_state(PlayerId::new(0), "life", 20);
    state.public.set_player_state(PlayerId::new(1), "life", 20);

    // Queue multiple effects
    for i in 1..=3 {
        let mut batch = EffectBatch::new();
        batch.add_player(Effect::damage(i), PlayerId::new(1));
        resolver.queue_action(Action::new(TemplateId::new(1)), batch, PlayerId::new(0));
    }

    let context = ResolverContext::simple(2);
    let status = resolver.process(&mut state, &context);

    assert_eq!(status, ResolutionStatus::Complete);
    // Total damage: 1 + 2 + 3 = 6
    assert_eq!(state.public.get_player_state(PlayerId::new(1), "life", 0), 14);
}

/// Test that immediate resolution has no priority player.
#[test]
fn test_immediate_no_priority() {
    let mut resolver = ImmediateResolution::new();

    let mut batch = EffectBatch::new();
    batch.add_player(Effect::damage(5), PlayerId::new(1));
    resolver.queue_action(Action::new(TemplateId::new(1)), batch, PlayerId::new(0));

    // Even with pending effects, no priority player
    assert_eq!(resolver.priority_player(), None);
}

// =============================================================================
// Priority Stack Tests
// =============================================================================

/// Test basic priority stack resolution.
#[test]
fn test_priority_stack_basic() {
    let mut stack = PriorityStack::new(2);
    let mut state = GameState::new(2, 42);
    state.public.set_player_state(PlayerId::new(1), "life", 20);

    // Queue damage
    let mut batch = EffectBatch::new();
    batch.add_player(Effect::damage(5), PlayerId::new(1));
    stack.queue_action(Action::new(TemplateId::new(1)), batch, PlayerId::new(0));

    let context = ResolverContext::simple(2);

    // Should wait for priority
    let status = stack.process(&mut state, &context);
    assert_eq!(status, ResolutionStatus::WaitingForPriority(PlayerId::new(0)));

    // Both players pass
    stack.pass(PlayerId::new(0));
    stack.pass(PlayerId::new(1));

    // Now should resolve
    let status = stack.process(&mut state, &context);
    assert_eq!(status, ResolutionStatus::Complete);
    assert_eq!(state.public.get_player_state(PlayerId::new(1), "life", 0), 15);
}

/// Test LIFO resolution order.
#[test]
fn test_priority_stack_lifo() {
    let mut stack = PriorityStack::new(2);
    let mut state = GameState::new(2, 42);
    state.public.set_player_state(PlayerId::new(0), "counter", 0);

    let context = ResolverContext::simple(2);

    // Push effect that sets counter to 1
    let mut batch1 = EffectBatch::new();
    batch1.add_player(Effect::set_player("counter", 1), PlayerId::new(0));
    stack.queue_action(Action::new(TemplateId::new(1)), batch1, PlayerId::new(0));

    // Respond with effect that sets counter to 2 (should resolve first)
    let mut batch2 = EffectBatch::new();
    batch2.add_player(Effect::set_player("counter", 2), PlayerId::new(0));
    stack.respond(batch2, PlayerId::new(1), "Response".to_string());

    assert_eq!(stack.stack_size(), 2);

    // Both pass - resolve top (sets to 2)
    stack.pass(PlayerId::new(1));
    stack.pass(PlayerId::new(0));
    let status = stack.process(&mut state, &context);
    assert_eq!(status, ResolutionStatus::Processing);
    assert_eq!(state.public.get_player_state(PlayerId::new(0), "counter", 0), 2);

    // Both pass again - resolve bottom (sets to 1)
    stack.pass(PlayerId::new(1));
    stack.pass(PlayerId::new(0));
    let status = stack.process(&mut state, &context);
    assert_eq!(status, ResolutionStatus::Complete);
    // Final value is 1 (last resolved)
    assert_eq!(state.public.get_player_state(PlayerId::new(0), "counter", 0), 1);
}

/// Test 4-player priority passing.
#[test]
fn test_priority_stack_four_player() {
    let mut stack = PriorityStack::new(4);
    let mut state = GameState::new(4, 42);
    state.public.set_player_state(PlayerId::new(0), "life", 20);

    // Queue effect
    let mut batch = EffectBatch::new();
    batch.add_player(Effect::damage(3), PlayerId::new(0));
    stack.queue_action(Action::new(TemplateId::new(1)), batch, PlayerId::new(0));

    let context = ResolverContext::simple(4);

    // Need all 4 players to pass
    assert_eq!(stack.priority_player(), Some(PlayerId::new(0)));

    stack.pass(PlayerId::new(0));
    assert_eq!(stack.priority_player(), Some(PlayerId::new(1)));

    stack.pass(PlayerId::new(1));
    assert_eq!(stack.priority_player(), Some(PlayerId::new(2)));

    stack.pass(PlayerId::new(2));
    assert_eq!(stack.priority_player(), Some(PlayerId::new(3)));

    stack.pass(PlayerId::new(3));

    // Now process should resolve
    let status = stack.process(&mut state, &context);
    assert_eq!(status, ResolutionStatus::Complete);
    assert_eq!(state.public.get_player_state(PlayerId::new(0), "life", 0), 17);
}

/// Test that responding resets passes.
#[test]
fn test_priority_respond_resets_passes() {
    let mut stack = PriorityStack::new(2);
    let mut state = GameState::new(2, 42);

    // Queue initial effect
    let mut batch = EffectBatch::new();
    batch.add_player(Effect::damage(5), PlayerId::new(1));
    stack.queue_action(Action::new(TemplateId::new(1)), batch, PlayerId::new(0));

    // Player 0 passes
    stack.pass(PlayerId::new(0));
    assert_eq!(stack.priority_player(), Some(PlayerId::new(1)));

    // Player 1 responds instead of passing
    let mut response = EffectBatch::new();
    response.add_player(Effect::heal(3), PlayerId::new(1));
    stack.respond(response, PlayerId::new(1), "Heal response".to_string());

    // Stack has 2 entries, priority back to responder
    assert_eq!(stack.stack_size(), 2);
    assert_eq!(stack.priority_player(), Some(PlayerId::new(1)));

    // Both players need to pass again
    let context = ResolverContext::simple(2);
    let status = stack.process(&mut state, &context);
    assert_eq!(status, ResolutionStatus::WaitingForPriority(PlayerId::new(1)));
}

// =============================================================================
// Triggered Effect Integration Tests
// =============================================================================

/// Test triggered effects with immediate resolution.
#[test]
fn test_immediate_with_triggers() {
    let mut resolver = ImmediateResolution::new();
    let mut state = GameState::new(2, 42);
    state.public.set_player_state(PlayerId::new(0), "life", 20);
    state.public.set_player_state(PlayerId::new(1), "life", 20);

    // Set up trigger registry
    let mut registry = TriggerRegistry::new();
    const DAMAGE_DEALT: EventTypeId = EventTypeId::new(1);

    // Register a trigger: when damage is dealt, heal 1
    let trigger = Trigger::new(TriggerId::new(1), "Lifesteal", DAMAGE_DEALT)
        .with_controller(PlayerId::new(0))
        .with_condition(TriggerCondition::Always)
        .with_effect(Effect::heal(1));
    registry.register_with_id(trigger);

    // Queue damage effect
    let mut batch = EffectBatch::new();
    batch.add_player(Effect::damage(5), PlayerId::new(1));
    resolver.queue_action(Action::new(TemplateId::new(1)), batch, PlayerId::new(0));

    let context = ResolverContext::simple(2);
    resolver.process(&mut state, &context);

    // Damage applied
    assert_eq!(state.public.get_player_state(PlayerId::new(1), "life", 0), 15);

    // Now fire the trigger event and queue the triggered effect
    let event = GameEvent::new(DAMAGE_DEALT).with_player(PlayerId::new(0));
    let triggered = registry.find_triggers(&event, &state, TriggerTiming::After, None);

    for t in triggered {
        resolver.queue_triggered(t);
    }

    resolver.process(&mut state, &context);

    // Heal applied to controller
    assert_eq!(state.public.get_player_state(PlayerId::new(0), "life", 0), 21);
}

/// Test triggered effects with priority stack.
#[test]
fn test_priority_with_triggers() {
    let mut stack = PriorityStack::new(2);
    let mut state = GameState::new(2, 42);
    state.public.set_player_state(PlayerId::new(0), "life", 20);
    state.public.set_player_state(PlayerId::new(1), "life", 20);

    // Set up trigger registry
    let mut registry = TriggerRegistry::new();
    const DAMAGE_DEALT: EventTypeId = EventTypeId::new(1);

    // Register a trigger: when damage dealt, controller heals 2
    // (Triggered effects target their controller by default)
    let trigger = Trigger::new(TriggerId::new(1), "Lifesteal", DAMAGE_DEALT)
        .with_controller(PlayerId::new(0))  // Attacker's trigger
        .with_condition(TriggerCondition::Always)
        .with_effect(Effect::heal(2));
    registry.register_with_id(trigger);

    // Queue initial damage (player 0 attacks player 1)
    let mut batch = EffectBatch::new();
    batch.add_player(Effect::damage(5), PlayerId::new(1));
    stack.queue_action(Action::new(TemplateId::new(1)), batch, PlayerId::new(0));

    let context = ResolverContext::simple(2);

    // Both pass
    stack.pass(PlayerId::new(0));
    stack.pass(PlayerId::new(1));

    // Resolve damage
    stack.process(&mut state, &context);
    assert_eq!(state.public.get_player_state(PlayerId::new(1), "life", 0), 15);

    // Fire trigger and queue
    let event = GameEvent::new(DAMAGE_DEALT).with_player(PlayerId::new(0));
    let triggered = registry.find_triggers(&event, &state, TriggerTiming::After, None);

    for t in triggered {
        stack.queue_triggered(t);
    }

    // Flush triggers to stack
    stack.process(&mut state, &context);

    // Now there's a triggered effect on the stack
    assert_eq!(stack.stack_size(), 1);

    // Both pass to resolve the triggered effect
    stack.pass(PlayerId::new(0));
    stack.pass(PlayerId::new(1));

    stack.process(&mut state, &context);

    // Lifesteal heals the controller (player 0)
    assert_eq!(state.public.get_player_state(PlayerId::new(0), "life", 0), 22);
}

// =============================================================================
// Stack Source Tests
// =============================================================================

/// Test that stack entries track their source correctly.
#[test]
fn test_stack_entry_source() {
    let mut stack = PriorityStack::new(2);

    // Queue an action
    let action = Action::new(TemplateId::new(42));
    let mut batch = EffectBatch::new();
    batch.add_player(Effect::damage(1), PlayerId::new(1));
    stack.queue_action(action.clone(), batch, PlayerId::new(0));

    let entry = stack.peek_top().unwrap();
    match &entry.source {
        StackSource::Action(a) => assert_eq!(a.template, TemplateId::new(42)),
        _ => panic!("Expected Action source"),
    }

    // Add a response
    let mut response = EffectBatch::new();
    response.add_player(Effect::heal(1), PlayerId::new(1));
    stack.respond(response, PlayerId::new(1), "Counter".to_string());

    let entry = stack.peek_top().unwrap();
    match &entry.source {
        StackSource::Response { description } => assert_eq!(description, "Counter"),
        _ => panic!("Expected Response source"),
    }
}

// =============================================================================
// Edge Cases
// =============================================================================

/// Test empty batch handling.
#[test]
fn test_empty_batch_not_queued() {
    let mut resolver = ImmediateResolution::new();
    let batch = EffectBatch::new();
    resolver.queue_action(Action::new(TemplateId::new(1)), batch, PlayerId::new(0));
    assert!(resolver.is_complete());

    let mut stack = PriorityStack::new(2);
    let batch = EffectBatch::new();
    stack.queue_action(Action::new(TemplateId::new(1)), batch, PlayerId::new(0));
    assert!(stack.is_complete());
}

/// Test clearing resolvers.
#[test]
fn test_clear_resolvers() {
    let mut resolver = ImmediateResolution::new();
    let mut batch = EffectBatch::new();
    batch.add_player(Effect::damage(5), PlayerId::new(1));
    resolver.queue_action(Action::new(TemplateId::new(1)), batch, PlayerId::new(0));
    assert!(!resolver.is_complete());
    resolver.clear();
    assert!(resolver.is_complete());

    let mut stack = PriorityStack::new(2);
    let mut batch = EffectBatch::new();
    batch.add_player(Effect::damage(5), PlayerId::new(1));
    stack.queue_action(Action::new(TemplateId::new(1)), batch, PlayerId::new(0));
    assert!(!stack.is_complete());
    stack.clear();
    assert!(stack.is_complete());
}

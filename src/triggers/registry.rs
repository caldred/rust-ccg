//! Trigger registry.
//!
//! The registry stores triggers and provides efficient lookup when events occur.
//! Games register triggers at startup or when cards enter play, and the registry
//! finds matching triggers for each event.

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::core::{EntityId, GameState, PlayerId};
use crate::effects::Effect;

use super::condition::{ConditionContext, ConditionEvaluator, TriggerCondition};
use super::event::{EventTypeId, GameEvent};

/// Unique identifier for a trigger.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TriggerId(pub u32);

impl TriggerId {
    /// Create a new trigger ID.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw ID value.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for TriggerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Trigger({})", self.0)
    }
}

/// When in the event resolution process the trigger fires.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TriggerTiming {
    /// Before the event resolves (can potentially modify/cancel).
    Before,
    /// After the event resolves (most common).
    #[default]
    After,
    /// Instead of the event (replacement effect).
    Instead,
}

/// A trigger definition.
///
/// Triggers watch for events and execute effects when conditions are met.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Trigger {
    /// Unique identifier.
    pub id: TriggerId,

    /// Human-readable name (for debugging).
    pub name: String,

    /// The source entity that owns this trigger.
    /// `None` for global/game triggers.
    pub source: Option<EntityId>,

    /// The controller of this trigger.
    /// Used to determine who makes choices for triggered effects.
    pub controller: Option<PlayerId>,

    /// Event types this trigger listens for.
    /// For efficient lookup, we index by event type.
    pub event_types: Vec<EventTypeId>,

    /// Additional conditions beyond event type.
    pub condition: TriggerCondition,

    /// When in resolution this trigger fires.
    pub timing: TriggerTiming,

    /// Effects to execute when triggered.
    pub effects: Vec<Effect>,

    /// Is this trigger currently active?
    pub enabled: bool,

    /// How many times can this trigger fire? `None` = unlimited.
    pub uses_remaining: Option<u32>,

    /// Priority for ordering triggers (higher fires first).
    /// When equal, triggers are ordered by ID for stability.
    pub priority: i32,
}

impl Trigger {
    /// Create a new trigger.
    pub fn new(id: TriggerId, name: impl Into<String>, event_type: EventTypeId) -> Self {
        Self {
            id,
            name: name.into(),
            source: None,
            controller: None,
            event_types: vec![event_type],
            condition: TriggerCondition::Always,
            timing: TriggerTiming::default(),
            effects: Vec::new(),
            enabled: true,
            uses_remaining: None,
            priority: 0,
        }
    }

    /// Set the source entity (builder pattern).
    #[must_use]
    pub fn with_source(mut self, source: EntityId) -> Self {
        self.source = Some(source);
        self
    }

    /// Set the controller (builder pattern).
    #[must_use]
    pub fn with_controller(mut self, controller: PlayerId) -> Self {
        self.controller = Some(controller);
        self
    }

    /// Add an event type to listen for (builder pattern).
    #[must_use]
    pub fn also_on(mut self, event_type: EventTypeId) -> Self {
        if !self.event_types.contains(&event_type) {
            self.event_types.push(event_type);
        }
        self
    }

    /// Set the condition (builder pattern).
    #[must_use]
    pub fn with_condition(mut self, condition: TriggerCondition) -> Self {
        self.condition = condition;
        self
    }

    /// Set the timing (builder pattern).
    #[must_use]
    pub fn with_timing(mut self, timing: TriggerTiming) -> Self {
        self.timing = timing;
        self
    }

    /// Add an effect (builder pattern).
    #[must_use]
    pub fn with_effect(mut self, effect: Effect) -> Self {
        self.effects.push(effect);
        self
    }

    /// Set limited uses (builder pattern).
    #[must_use]
    pub fn with_uses(mut self, uses: u32) -> Self {
        self.uses_remaining = Some(uses);
        self
    }

    /// Set priority (builder pattern).
    /// Higher priority triggers fire first.
    #[must_use]
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Check if this trigger can fire (enabled and has uses).
    #[must_use]
    pub fn can_fire(&self) -> bool {
        self.enabled && self.uses_remaining.is_none_or(|u| u > 0)
    }

    /// Consume one use of this trigger.
    pub fn use_trigger(&mut self) {
        if let Some(ref mut uses) = self.uses_remaining {
            *uses = uses.saturating_sub(1);
        }
    }
}

/// A triggered effect ready to be resolved.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TriggeredEffect {
    /// The trigger that fired.
    pub trigger_id: TriggerId,

    /// The controller (who makes choices).
    pub controller: Option<PlayerId>,

    /// The source entity.
    pub source: Option<EntityId>,

    /// Effects to execute.
    pub effects: Vec<Effect>,

    /// The event that caused this trigger.
    pub triggering_event: GameEvent,

    /// When this trigger fires (for replacement effect detection).
    pub timing: TriggerTiming,
}

/// Registry for triggers.
///
/// The registry stores triggers and provides efficient lookup by event type.
/// When an event fires, use `find_triggers` to get matching triggers.
#[derive(Clone, Debug, Default)]
pub struct TriggerRegistry {
    /// All registered triggers.
    triggers: FxHashMap<TriggerId, Trigger>,

    /// Index by event type for fast lookup.
    by_event_type: FxHashMap<EventTypeId, Vec<TriggerId>>,

    /// Next trigger ID to allocate.
    next_id: u32,
}

impl TriggerRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a trigger, returns its ID.
    pub fn register(&mut self, mut trigger: Trigger) -> TriggerId {
        // Assign ID if not set
        if trigger.id.0 == 0 {
            trigger.id = TriggerId::new(self.next_id);
            self.next_id += 1;
        }

        let id = trigger.id;

        // Index by event types
        for event_type in &trigger.event_types {
            self.by_event_type
                .entry(*event_type)
                .or_default()
                .push(id);
        }

        self.triggers.insert(id, trigger);
        id
    }

    /// Register a trigger with a specific ID.
    pub fn register_with_id(&mut self, trigger: Trigger) -> TriggerId {
        let id = trigger.id;

        // Update next_id if needed
        if id.0 >= self.next_id {
            self.next_id = id.0 + 1;
        }

        // Index by event types
        for event_type in &trigger.event_types {
            self.by_event_type
                .entry(*event_type)
                .or_default()
                .push(id);
        }

        self.triggers.insert(id, trigger);
        id
    }

    /// Unregister a trigger.
    pub fn unregister(&mut self, id: TriggerId) -> Option<Trigger> {
        if let Some(trigger) = self.triggers.remove(&id) {
            // Remove from event type index and track empty entries
            let mut empty_types = Vec::new();
            for event_type in &trigger.event_types {
                if let Some(list) = self.by_event_type.get_mut(event_type) {
                    list.retain(|&tid| tid != id);
                    if list.is_empty() {
                        empty_types.push(*event_type);
                    }
                }
            }
            // Clean up empty vectors
            for event_type in empty_types {
                self.by_event_type.remove(&event_type);
            }
            Some(trigger)
        } else {
            None
        }
    }

    /// Get a trigger by ID.
    #[must_use]
    pub fn get(&self, id: TriggerId) -> Option<&Trigger> {
        self.triggers.get(&id)
    }

    /// Get a mutable trigger by ID.
    pub fn get_mut(&mut self, id: TriggerId) -> Option<&mut Trigger> {
        self.triggers.get_mut(&id)
    }

    /// Find all triggers that should fire for an event.
    ///
    /// Returns triggered effects ready for resolution, sorted by priority
    /// (higher priority first), then by trigger ID for stability.
    pub fn find_triggers(
        &self,
        event: &GameEvent,
        state: &GameState,
        timing: TriggerTiming,
        custom_eval: Option<&dyn Fn(&str, &GameEvent, &GameState) -> bool>,
    ) -> Vec<TriggeredEffect> {
        // Store (priority, trigger_id, effect) for sorting
        let mut results: Vec<(i32, TriggerId, TriggeredEffect)> = Vec::new();

        // Get triggers registered for this event type
        let Some(trigger_ids) = self.by_event_type.get(&event.event_type) else {
            return Vec::new();
        };

        let ctx = if let Some(eval) = custom_eval {
            ConditionContext::new(event, state).with_custom_eval(eval)
        } else {
            ConditionContext::new(event, state)
        };

        for &trigger_id in trigger_ids {
            let Some(trigger) = self.triggers.get(&trigger_id) else {
                continue;
            };

            // Check timing
            if trigger.timing != timing {
                continue;
            }

            // Check if can fire
            if !trigger.can_fire() {
                continue;
            }

            // Check condition
            if !ConditionEvaluator::evaluate(&trigger.condition, &ctx) {
                continue;
            }

            results.push((trigger.priority, trigger_id, TriggeredEffect {
                trigger_id,
                controller: trigger.controller,
                source: trigger.source,
                effects: trigger.effects.clone(),
                triggering_event: event.clone(),
                timing: trigger.timing,
            }));
        }

        // Sort by priority (descending), then by trigger_id (ascending) for stability
        results.sort_by(|a, b| {
            b.0.cmp(&a.0).then_with(|| a.1.0.cmp(&b.1.0))
        });

        results.into_iter().map(|(_, _, effect)| effect).collect()
    }

    /// Find triggers for a specific source entity.
    pub fn triggers_for_source(&self, source: EntityId) -> Vec<&Trigger> {
        self.triggers
            .values()
            .filter(|t| t.source == Some(source))
            .collect()
    }

    /// Remove all triggers owned by a source entity.
    pub fn remove_for_source(&mut self, source: EntityId) {
        let to_remove: Vec<_> = self
            .triggers
            .iter()
            .filter(|(_, t)| t.source == Some(source))
            .map(|(&id, _)| id)
            .collect();

        for id in to_remove {
            self.unregister(id);
        }
    }

    /// Enable or disable a trigger.
    pub fn set_enabled(&mut self, id: TriggerId, enabled: bool) {
        if let Some(trigger) = self.triggers.get_mut(&id) {
            trigger.enabled = enabled;
        }
    }

    /// Get total trigger count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.triggers.len()
    }

    /// Check if registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.triggers.is_empty()
    }

    /// Iterate all triggers.
    pub fn iter(&self) -> impl Iterator<Item = &Trigger> {
        self.triggers.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_state() -> GameState {
        GameState::new(2, 42)
    }

    #[test]
    fn test_trigger_id() {
        let id = TriggerId::new(5);
        assert_eq!(id.raw(), 5);
        assert_eq!(format!("{}", id), "Trigger(5)");
    }

    #[test]
    fn test_trigger_builder() {
        let trigger = Trigger::new(TriggerId::new(1), "DamageResponse", EventTypeId::new(1))
            .with_source(EntityId(10))
            .with_controller(PlayerId::new(0))
            .with_condition(TriggerCondition::ValueAtLeast { index: 0, min: 3 })
            .with_timing(TriggerTiming::After)
            .with_effect(Effect::damage(2))
            .with_uses(2);

        assert_eq!(trigger.id, TriggerId::new(1));
        assert_eq!(trigger.source, Some(EntityId(10)));
        assert_eq!(trigger.controller, Some(PlayerId::new(0)));
        assert!(matches!(trigger.timing, TriggerTiming::After));
        assert_eq!(trigger.uses_remaining, Some(2));
        assert!(trigger.can_fire());
    }

    #[test]
    fn test_trigger_uses() {
        let mut trigger = Trigger::new(TriggerId::new(1), "Limited", EventTypeId::new(1))
            .with_uses(2);

        assert!(trigger.can_fire());
        trigger.use_trigger();
        assert!(trigger.can_fire());
        assert_eq!(trigger.uses_remaining, Some(1));
        trigger.use_trigger();
        assert!(!trigger.can_fire());
        assert_eq!(trigger.uses_remaining, Some(0));
    }

    #[test]
    fn test_registry_register() {
        let mut registry = TriggerRegistry::new();

        let trigger = Trigger::new(TriggerId::new(0), "Test", EventTypeId::new(1));
        let id = registry.register(trigger);

        assert!(registry.get(id).is_some());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_registry_unregister() {
        let mut registry = TriggerRegistry::new();

        let trigger = Trigger::new(TriggerId::new(1), "Test", EventTypeId::new(1));
        let id = registry.register_with_id(trigger);

        assert_eq!(registry.len(), 1);
        let removed = registry.unregister(id);
        assert!(removed.is_some());
        assert_eq!(registry.len(), 0);
        assert!(registry.get(id).is_none());
    }

    #[test]
    fn test_find_triggers_basic() {
        let state = test_state();
        let mut registry = TriggerRegistry::new();

        // Register a simple trigger
        let trigger = Trigger::new(TriggerId::new(1), "OnDamage", EventTypeId::new(1))
            .with_effect(Effect::heal(1));
        registry.register_with_id(trigger);

        // Create matching event
        let event = GameEvent::new(EventTypeId::new(1));
        let results = registry.find_triggers(&event, &state, TriggerTiming::After, None);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].trigger_id, TriggerId::new(1));
    }

    #[test]
    fn test_find_triggers_with_condition() {
        let state = test_state();
        let mut registry = TriggerRegistry::new();

        // Register trigger with value condition
        let trigger = Trigger::new(TriggerId::new(1), "BigDamage", EventTypeId::new(1))
            .with_condition(TriggerCondition::ValueAtLeast { index: 0, min: 5 });
        registry.register_with_id(trigger);

        // Small damage - shouldn't trigger
        let small_event = GameEvent::new(EventTypeId::new(1)).with_value(3);
        let results = registry.find_triggers(&small_event, &state, TriggerTiming::After, None);
        assert_eq!(results.len(), 0);

        // Big damage - should trigger
        let big_event = GameEvent::new(EventTypeId::new(1)).with_value(7);
        let results = registry.find_triggers(&big_event, &state, TriggerTiming::After, None);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_find_triggers_timing() {
        let state = test_state();
        let mut registry = TriggerRegistry::new();

        // Register before and after triggers
        let before = Trigger::new(TriggerId::new(1), "Before", EventTypeId::new(1))
            .with_timing(TriggerTiming::Before);
        let after = Trigger::new(TriggerId::new(2), "After", EventTypeId::new(1))
            .with_timing(TriggerTiming::After);

        registry.register_with_id(before);
        registry.register_with_id(after);

        let event = GameEvent::new(EventTypeId::new(1));

        let before_results = registry.find_triggers(&event, &state, TriggerTiming::Before, None);
        assert_eq!(before_results.len(), 1);
        assert_eq!(before_results[0].trigger_id, TriggerId::new(1));

        let after_results = registry.find_triggers(&event, &state, TriggerTiming::After, None);
        assert_eq!(after_results.len(), 1);
        assert_eq!(after_results[0].trigger_id, TriggerId::new(2));
    }

    #[test]
    fn test_find_triggers_disabled() {
        let state = test_state();
        let mut registry = TriggerRegistry::new();

        let trigger = Trigger::new(TriggerId::new(1), "Test", EventTypeId::new(1));
        let id = registry.register_with_id(trigger);

        let event = GameEvent::new(EventTypeId::new(1));

        // Should find when enabled
        let results = registry.find_triggers(&event, &state, TriggerTiming::After, None);
        assert_eq!(results.len(), 1);

        // Disable and check again
        registry.set_enabled(id, false);
        let results = registry.find_triggers(&event, &state, TriggerTiming::After, None);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_triggers_for_source() {
        let mut registry = TriggerRegistry::new();

        let t1 = Trigger::new(TriggerId::new(1), "Card1", EventTypeId::new(1))
            .with_source(EntityId(10));
        let t2 = Trigger::new(TriggerId::new(2), "Card1Also", EventTypeId::new(2))
            .with_source(EntityId(10));
        let t3 = Trigger::new(TriggerId::new(3), "Card2", EventTypeId::new(1))
            .with_source(EntityId(20));

        registry.register_with_id(t1);
        registry.register_with_id(t2);
        registry.register_with_id(t3);

        let for_10 = registry.triggers_for_source(EntityId(10));
        assert_eq!(for_10.len(), 2);

        let for_20 = registry.triggers_for_source(EntityId(20));
        assert_eq!(for_20.len(), 1);
    }

    #[test]
    fn test_remove_for_source() {
        let mut registry = TriggerRegistry::new();

        let t1 = Trigger::new(TriggerId::new(1), "Card1", EventTypeId::new(1))
            .with_source(EntityId(10));
        let t2 = Trigger::new(TriggerId::new(2), "Card2", EventTypeId::new(1))
            .with_source(EntityId(20));

        registry.register_with_id(t1);
        registry.register_with_id(t2);

        assert_eq!(registry.len(), 2);

        registry.remove_for_source(EntityId(10));

        assert_eq!(registry.len(), 1);
        assert!(registry.get(TriggerId::new(1)).is_none());
        assert!(registry.get(TriggerId::new(2)).is_some());
    }

    #[test]
    fn test_multiple_event_types() {
        let state = test_state();
        let mut registry = TriggerRegistry::new();

        // Trigger that fires on multiple event types
        let trigger = Trigger::new(TriggerId::new(1), "Multi", EventTypeId::new(1))
            .also_on(EventTypeId::new(2))
            .also_on(EventTypeId::new(3));
        registry.register_with_id(trigger);

        // Should fire on all three event types
        for event_type_id in [1, 2, 3] {
            let event = GameEvent::new(EventTypeId::new(event_type_id));
            let results = registry.find_triggers(&event, &state, TriggerTiming::After, None);
            assert_eq!(results.len(), 1, "Should trigger on event type {}", event_type_id);
        }

        // Should not fire on other event types
        let event = GameEvent::new(EventTypeId::new(99));
        let results = registry.find_triggers(&event, &state, TriggerTiming::After, None);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_n_player_triggers() {
        let state = GameState::new(4, 42);
        let mut registry = TriggerRegistry::new();

        // Register triggers for each player
        for player_idx in 0..4u8 {
            let trigger = Trigger::new(
                TriggerId::new(player_idx as u32),
                format!("Player{}", player_idx),
                EventTypeId::new(1),
            )
            .with_controller(PlayerId::new(player_idx))
            .with_condition(TriggerCondition::ForPlayer(PlayerId::new(player_idx)));

            registry.register_with_id(trigger);
        }

        // Each player's event should only trigger their trigger
        for player_idx in 0..4u8 {
            let event = GameEvent::for_player(EventTypeId::new(1), PlayerId::new(player_idx));
            let results = registry.find_triggers(&event, &state, TriggerTiming::After, None);

            assert_eq!(results.len(), 1);
            assert_eq!(results[0].controller, Some(PlayerId::new(player_idx)));
        }
    }
}

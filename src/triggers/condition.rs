//! Trigger conditions.
//!
//! Conditions determine when a trigger fires based on event data.
//! The engine provides common condition types; games can use custom
//! conditions for game-specific logic.

use serde::{Deserialize, Serialize};

use crate::core::{EntityId, GameState, PlayerId, ZoneId};

use super::event::{EventTypeId, GameEvent};

/// A condition that must be met for a trigger to fire.
///
/// Conditions are checked against the event and game state to determine
/// if the trigger should activate.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TriggerCondition {
    // === Event Type Matching ===

    /// Match a specific event type.
    EventType(EventTypeId),

    /// Match any of the specified event types.
    AnyEventType(Vec<EventTypeId>),

    // === Entity Filters ===

    /// Source must be a specific entity.
    SourceIs(EntityId),

    /// Target must be a specific entity.
    TargetIs(EntityId),

    /// Source must be controlled by specified player.
    SourceControlledBy(PlayerId),

    /// Target must be controlled by specified player.
    TargetControlledBy(PlayerId),

    /// Event must be for specified player.
    ForPlayer(PlayerId),

    /// Source must be in specified zone.
    SourceInZone(ZoneId),

    /// Target must be in specified zone.
    TargetInZone(ZoneId),

    // === Value Filters ===

    /// Value at index must be at least N.
    ValueAtLeast { index: usize, min: i64 },

    /// Value at index must be at most N.
    ValueAtMost { index: usize, max: i64 },

    /// Value at index must be in range [min, max].
    ValueInRange { index: usize, min: i64, max: i64 },

    // === Tag Filters ===

    /// Event must have specified tag.
    HasTag(String),

    /// Event must not have specified tag.
    NotTag(String),

    // === Combinators ===

    /// All conditions must be true.
    All(Vec<TriggerCondition>),

    /// At least one condition must be true.
    Any(Vec<TriggerCondition>),

    /// Condition must be false.
    Not(Box<TriggerCondition>),

    // === Special ===

    /// Always matches (no filter).
    Always,

    /// Never matches (disabled trigger).
    Never,

    /// Custom condition (evaluated by game-specific code).
    Custom(String),
}

impl TriggerCondition {
    /// Create an event type condition.
    pub fn event(event_type: EventTypeId) -> Self {
        Self::EventType(event_type)
    }

    /// Create a condition for events involving a specific player.
    pub fn for_player(player: PlayerId) -> Self {
        Self::ForPlayer(player)
    }

    /// Create a condition requiring source controlled by player.
    pub fn source_controlled_by(player: PlayerId) -> Self {
        Self::SourceControlledBy(player)
    }

    /// Create a condition requiring target controlled by player.
    pub fn target_controlled_by(player: PlayerId) -> Self {
        Self::TargetControlledBy(player)
    }

    /// Create a minimum value condition.
    pub fn value_at_least(index: usize, min: i64) -> Self {
        Self::ValueAtLeast { index, min }
    }

    /// Create an AND condition.
    pub fn all(conditions: impl IntoIterator<Item = TriggerCondition>) -> Self {
        Self::All(conditions.into_iter().collect())
    }

    /// Create an OR condition.
    pub fn any(conditions: impl IntoIterator<Item = TriggerCondition>) -> Self {
        Self::Any(conditions.into_iter().collect())
    }

    /// Negate this condition.
    pub fn negate(self) -> Self {
        Self::Not(Box::new(self))
    }

    /// Add another condition with AND.
    pub fn and(self, other: TriggerCondition) -> Self {
        match self {
            Self::All(mut conditions) => {
                conditions.push(other);
                Self::All(conditions)
            }
            _ => Self::All(vec![self, other]),
        }
    }

    /// Add another condition with OR.
    pub fn or(self, other: TriggerCondition) -> Self {
        match self {
            Self::Any(mut conditions) => {
                conditions.push(other);
                Self::Any(conditions)
            }
            _ => Self::Any(vec![self, other]),
        }
    }
}

/// Context for evaluating trigger conditions.
pub struct ConditionContext<'a> {
    /// The event being checked.
    pub event: &'a GameEvent,
    /// Current game state.
    pub state: &'a GameState,
    /// Custom condition evaluator (provided by game).
    pub eval_custom: Option<&'a dyn Fn(&str, &GameEvent, &GameState) -> bool>,
}

impl<'a> ConditionContext<'a> {
    /// Create a new context.
    pub fn new(event: &'a GameEvent, state: &'a GameState) -> Self {
        Self {
            event,
            state,
            eval_custom: None,
        }
    }

    /// Add a custom condition evaluator.
    pub fn with_custom_eval(
        mut self,
        eval: &'a dyn Fn(&str, &GameEvent, &GameState) -> bool,
    ) -> Self {
        self.eval_custom = Some(eval);
        self
    }
}

/// Evaluator for trigger conditions.
pub struct ConditionEvaluator;

impl ConditionEvaluator {
    /// Check if a condition is satisfied.
    pub fn evaluate(condition: &TriggerCondition, ctx: &ConditionContext) -> bool {
        match condition {
            TriggerCondition::EventType(expected) => ctx.event.event_type == *expected,

            TriggerCondition::AnyEventType(types) => types.contains(&ctx.event.event_type),

            TriggerCondition::SourceIs(entity) => ctx.event.source == Some(*entity),

            TriggerCondition::TargetIs(entity) => ctx.event.target == Some(*entity),

            TriggerCondition::SourceControlledBy(player) => {
                if let Some(source) = ctx.event.source {
                    ctx.state
                        .get_card(source)
                        .is_some_and(|card| card.controller == Some(*player))
                } else {
                    false
                }
            }

            TriggerCondition::TargetControlledBy(player) => {
                if let Some(target) = ctx.event.target {
                    ctx.state
                        .get_card(target)
                        .is_some_and(|card| card.controller == Some(*player))
                } else {
                    false
                }
            }

            TriggerCondition::ForPlayer(player) => ctx.event.player == Some(*player),

            TriggerCondition::SourceInZone(zone) => {
                if let Some(source) = ctx.event.source {
                    ctx.state.zones.is_in_zone(source, *zone)
                } else {
                    false
                }
            }

            TriggerCondition::TargetInZone(zone) => {
                if let Some(target) = ctx.event.target {
                    ctx.state.zones.is_in_zone(target, *zone)
                } else {
                    false
                }
            }

            TriggerCondition::ValueAtLeast { index, min } => {
                ctx.event.value(*index, i64::MIN) >= *min
            }

            TriggerCondition::ValueAtMost { index, max } => {
                ctx.event.value(*index, i64::MAX) <= *max
            }

            TriggerCondition::ValueInRange { index, min, max } => {
                let value = ctx.event.value(*index, 0);
                value >= *min && value <= *max
            }

            TriggerCondition::HasTag(tag) => ctx.event.has_tag(tag),

            TriggerCondition::NotTag(tag) => !ctx.event.has_tag(tag),

            TriggerCondition::All(conditions) => {
                conditions.iter().all(|c| Self::evaluate(c, ctx))
            }

            TriggerCondition::Any(conditions) => {
                conditions.iter().any(|c| Self::evaluate(c, ctx))
            }

            TriggerCondition::Not(inner) => !Self::evaluate(inner, ctx),

            TriggerCondition::Always => true,

            TriggerCondition::Never => false,

            TriggerCondition::Custom(key) => {
                if let Some(eval) = ctx.eval_custom {
                    eval(key, ctx.event, ctx.state)
                } else {
                    // No evaluator provided, custom conditions fail
                    false
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::{CardId, CardInstance};

    fn test_state() -> GameState {
        GameState::new(2, 42)
    }

    #[test]
    fn test_event_type_condition() {
        let state = test_state();
        let event = GameEvent::new(EventTypeId::new(5));
        let ctx = ConditionContext::new(&event, &state);

        assert!(ConditionEvaluator::evaluate(
            &TriggerCondition::EventType(EventTypeId::new(5)),
            &ctx
        ));
        assert!(!ConditionEvaluator::evaluate(
            &TriggerCondition::EventType(EventTypeId::new(10)),
            &ctx
        ));
    }

    #[test]
    fn test_any_event_type_condition() {
        let state = test_state();
        let event = GameEvent::new(EventTypeId::new(5));
        let ctx = ConditionContext::new(&event, &state);

        let condition = TriggerCondition::AnyEventType(vec![
            EventTypeId::new(1),
            EventTypeId::new(5),
            EventTypeId::new(10),
        ]);

        assert!(ConditionEvaluator::evaluate(&condition, &ctx));

        let non_matching = TriggerCondition::AnyEventType(vec![
            EventTypeId::new(1),
            EventTypeId::new(2),
        ]);
        assert!(!ConditionEvaluator::evaluate(&non_matching, &ctx));
    }

    #[test]
    fn test_source_target_conditions() {
        let state = test_state();
        let event = GameEvent::new(EventTypeId::new(1))
            .with_source(EntityId(10))
            .with_target(EntityId(20));
        let ctx = ConditionContext::new(&event, &state);

        assert!(ConditionEvaluator::evaluate(
            &TriggerCondition::SourceIs(EntityId(10)),
            &ctx
        ));
        assert!(!ConditionEvaluator::evaluate(
            &TriggerCondition::SourceIs(EntityId(20)),
            &ctx
        ));
        assert!(ConditionEvaluator::evaluate(
            &TriggerCondition::TargetIs(EntityId(20)),
            &ctx
        ));
    }

    #[test]
    fn test_player_condition() {
        let state = test_state();
        let event = GameEvent::for_player(EventTypeId::new(1), PlayerId::new(1));
        let ctx = ConditionContext::new(&event, &state);

        assert!(ConditionEvaluator::evaluate(
            &TriggerCondition::ForPlayer(PlayerId::new(1)),
            &ctx
        ));
        assert!(!ConditionEvaluator::evaluate(
            &TriggerCondition::ForPlayer(PlayerId::new(0)),
            &ctx
        ));
    }

    #[test]
    fn test_value_conditions() {
        let state = test_state();
        let event = GameEvent::new(EventTypeId::new(1))
            .with_value(5);
        let ctx = ConditionContext::new(&event, &state);

        assert!(ConditionEvaluator::evaluate(
            &TriggerCondition::ValueAtLeast { index: 0, min: 3 },
            &ctx
        ));
        assert!(ConditionEvaluator::evaluate(
            &TriggerCondition::ValueAtLeast { index: 0, min: 5 },
            &ctx
        ));
        assert!(!ConditionEvaluator::evaluate(
            &TriggerCondition::ValueAtLeast { index: 0, min: 6 },
            &ctx
        ));

        assert!(ConditionEvaluator::evaluate(
            &TriggerCondition::ValueAtMost { index: 0, max: 10 },
            &ctx
        ));
        assert!(!ConditionEvaluator::evaluate(
            &TriggerCondition::ValueAtMost { index: 0, max: 4 },
            &ctx
        ));

        assert!(ConditionEvaluator::evaluate(
            &TriggerCondition::ValueInRange { index: 0, min: 3, max: 7 },
            &ctx
        ));
    }

    #[test]
    fn test_tag_conditions() {
        let state = test_state();
        let event = GameEvent::new(EventTypeId::new(1))
            .with_tag("combat")
            .with_tag("direct");
        let ctx = ConditionContext::new(&event, &state);

        assert!(ConditionEvaluator::evaluate(
            &TriggerCondition::HasTag("combat".to_string()),
            &ctx
        ));
        assert!(!ConditionEvaluator::evaluate(
            &TriggerCondition::HasTag("spell".to_string()),
            &ctx
        ));
        assert!(ConditionEvaluator::evaluate(
            &TriggerCondition::NotTag("spell".to_string()),
            &ctx
        ));
    }

    #[test]
    fn test_combinators() {
        let state = test_state();
        let event = GameEvent::new(EventTypeId::new(1))
            .with_value(5)
            .with_tag("combat");
        let ctx = ConditionContext::new(&event, &state);

        // All
        let all_true = TriggerCondition::All(vec![
            TriggerCondition::EventType(EventTypeId::new(1)),
            TriggerCondition::HasTag("combat".to_string()),
        ]);
        assert!(ConditionEvaluator::evaluate(&all_true, &ctx));

        let all_mixed = TriggerCondition::All(vec![
            TriggerCondition::EventType(EventTypeId::new(1)),
            TriggerCondition::HasTag("spell".to_string()),
        ]);
        assert!(!ConditionEvaluator::evaluate(&all_mixed, &ctx));

        // Any
        let any_true = TriggerCondition::Any(vec![
            TriggerCondition::EventType(EventTypeId::new(99)),
            TriggerCondition::HasTag("combat".to_string()),
        ]);
        assert!(ConditionEvaluator::evaluate(&any_true, &ctx));

        // Not
        let negated = TriggerCondition::Not(Box::new(
            TriggerCondition::EventType(EventTypeId::new(99))
        ));
        assert!(ConditionEvaluator::evaluate(&negated, &ctx));
    }

    #[test]
    fn test_always_never() {
        let state = test_state();
        let event = GameEvent::new(EventTypeId::new(1));
        let ctx = ConditionContext::new(&event, &state);

        assert!(ConditionEvaluator::evaluate(&TriggerCondition::Always, &ctx));
        assert!(!ConditionEvaluator::evaluate(&TriggerCondition::Never, &ctx));
    }

    #[test]
    fn test_custom_condition() {
        let state = test_state();
        let event = GameEvent::new(EventTypeId::new(1)).with_value(10);

        // Without evaluator
        let ctx = ConditionContext::new(&event, &state);
        assert!(!ConditionEvaluator::evaluate(
            &TriggerCondition::Custom("test".to_string()),
            &ctx
        ));

        // With evaluator
        let eval = |key: &str, event: &GameEvent, _state: &GameState| -> bool {
            key == "big_damage" && event.value(0, 0) >= 5
        };
        let ctx_with_eval = ConditionContext::new(&event, &state)
            .with_custom_eval(&eval);

        assert!(ConditionEvaluator::evaluate(
            &TriggerCondition::Custom("big_damage".to_string()),
            &ctx_with_eval
        ));
        assert!(!ConditionEvaluator::evaluate(
            &TriggerCondition::Custom("small_damage".to_string()),
            &ctx_with_eval
        ));
    }

    #[test]
    fn test_builder_methods() {
        let condition = TriggerCondition::event(EventTypeId::new(1))
            .and(TriggerCondition::for_player(PlayerId::new(0)))
            .and(TriggerCondition::value_at_least(0, 3));

        // Should create an All with 3 conditions
        if let TriggerCondition::All(conditions) = condition {
            assert_eq!(conditions.len(), 3);
        } else {
            panic!("Expected All condition");
        }
    }

    #[test]
    fn test_controlled_by_conditions() {
        let mut state = test_state();
        let battlefield = ZoneId::new(0);

        // Create a card controlled by player 1
        let card_entity = state.alloc_entity();
        let mut card = CardInstance::new(card_entity, CardId::new(1), PlayerId::new(0), battlefield);
        card.controller = Some(PlayerId::new(1));
        state.add_card(card);

        let event = GameEvent::new(EventTypeId::new(1))
            .with_source(card_entity);
        let ctx = ConditionContext::new(&event, &state);

        assert!(ConditionEvaluator::evaluate(
            &TriggerCondition::SourceControlledBy(PlayerId::new(1)),
            &ctx
        ));
        assert!(!ConditionEvaluator::evaluate(
            &TriggerCondition::SourceControlledBy(PlayerId::new(0)),
            &ctx
        ));
    }

    #[test]
    fn test_condition_serialization() {
        let condition = TriggerCondition::All(vec![
            TriggerCondition::EventType(EventTypeId::new(1)),
            TriggerCondition::ValueAtLeast { index: 0, min: 5 },
        ]);

        let json = serde_json::to_string(&condition).unwrap();
        let deserialized: TriggerCondition = serde_json::from_str(&json).unwrap();
        assert_eq!(condition, deserialized);
    }
}

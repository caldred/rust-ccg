//! Trigger system for event-driven abilities.
//!
//! Triggers allow cards and abilities to respond to game events.
//! The system is game-agnostic - games define their own event types
//! and register triggers accordingly.
//!
//! ## Key Components
//!
//! - [`EventTypeId`]: Opaque identifier for event types (game-defined)
//! - [`GameEvent`]: An event that occurred with contextual data
//! - [`TriggerCondition`]: Rules for when a trigger fires
//! - [`Trigger`]: A complete trigger definition
//! - [`TriggerRegistry`]: Storage and lookup for triggers
//!
//! ## Design Philosophy
//!
//! Like zones and templates, event types are not hardcoded. Games define
//! their own event types (damage dealt, card drawn, turn started, etc.)
//! and fire events at appropriate times. The engine provides the
//! infrastructure for matching triggers to events.
//!
//! ## Example Usage
//!
//! ```
//! use rust_ccg::core::{EntityId, GameState, PlayerId};
//! use rust_ccg::triggers::{
//!     EventTypeId, GameEvent, Trigger, TriggerCondition,
//!     TriggerRegistry, TriggerTiming, TriggerId,
//! };
//! use rust_ccg::effects::Effect;
//!
//! // Define event types (games typically do this at startup)
//! const DAMAGE_DEALT: EventTypeId = EventTypeId::new(1);
//! const CARD_PLAYED: EventTypeId = EventTypeId::new(2);
//!
//! // Create a registry
//! let mut registry = TriggerRegistry::new();
//!
//! // Register a trigger: "When this creature deals damage, gain 1 life"
//! let lifelink = Trigger::new(TriggerId::new(1), "Lifelink", DAMAGE_DEALT)
//!     .with_source(EntityId(10))  // The creature with lifelink
//!     .with_controller(PlayerId::new(0))
//!     .with_condition(TriggerCondition::SourceIs(EntityId(10)))
//!     .with_effect(Effect::heal(1));
//!
//! registry.register_with_id(lifelink);
//!
//! // When damage is dealt, find matching triggers
//! let state = GameState::new(2, 42);
//! let event = GameEvent::damage(DAMAGE_DEALT, EntityId(10), EntityId(20), 3);
//!
//! let triggered = registry.find_triggers(&event, &state, TriggerTiming::After, None);
//! assert_eq!(triggered.len(), 1);
//! ```
//!
//! ## N-Player Support
//!
//! The trigger system fully supports N players. Triggers can be conditioned
//! on specific players, and events carry player information for filtering.

mod condition;
mod event;
mod registry;

pub use condition::{ConditionContext, ConditionEvaluator, TriggerCondition};
pub use event::{EventTypeConfig, EventTypeId, GameEvent};
pub use registry::{Trigger, TriggerId, TriggerRegistry, TriggerTiming, TriggeredEffect};

//! Game event types.
//!
//! Events represent things that happen during a game. The engine provides
//! the event infrastructure; games define what events exist via `EventTypeId`.
//!
//! ## Design Philosophy
//!
//! Like zones and templates, event types are game-defined, not hardcoded.
//! The engine doesn't know about "damage dealt" or "card drawn" - games
//! register these event types and fire them appropriately.

use serde::{Deserialize, Serialize};

use crate::core::{EntityId, PlayerId, ZoneId};

/// Event type identifier. Games define what event types exist.
///
/// The engine doesn't interpret these - they're opaque identifiers.
/// Games assign meaning via configuration.
///
/// ## Example Event Types
///
/// A typical card game might define:
/// - `DAMAGE_DEALT` - When damage is dealt to an entity
/// - `CARD_DRAWN` - When a card is drawn
/// - `CARD_PLAYED` - When a card is played
/// - `TURN_START` - When a turn begins
/// - `TURN_END` - When a turn ends
/// - `ZONE_CHANGE` - When a card changes zones
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EventTypeId(pub u32);

impl EventTypeId {
    /// Create a new event type ID.
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

impl std::fmt::Display for EventTypeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EventType({})", self.0)
    }
}

/// A game event with contextual data.
///
/// Events are fired when things happen in the game. Triggers listen for
/// specific event types and may respond with effects.
///
/// ## Event Data
///
/// Events carry contextual information:
/// - `event_type`: What kind of event this is
/// - `source`: The entity that caused the event (if any)
/// - `target`: The entity affected by the event (if any)
/// - `player`: The player associated with the event (if any)
/// - `values`: Numeric values (damage amount, cards drawn, etc.)
/// - `zones`: Zone information (source zone, destination zone)
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GameEvent {
    /// The type of event.
    pub event_type: EventTypeId,

    /// The entity that caused/initiated the event.
    pub source: Option<EntityId>,

    /// The entity that was affected by the event.
    pub target: Option<EntityId>,

    /// The player associated with the event.
    pub player: Option<PlayerId>,

    /// Additional entities involved in the event.
    pub other_entities: Vec<EntityId>,

    /// Numeric values associated with the event.
    /// Games define the meaning of each index.
    pub values: Vec<i64>,

    /// Zone information (source zone, destination zone, etc.).
    pub zones: Vec<ZoneId>,

    /// String keys for custom event data.
    /// Used for game-specific filtering.
    pub tags: Vec<String>,
}

impl GameEvent {
    /// Create a new event with just a type.
    pub fn new(event_type: EventTypeId) -> Self {
        Self {
            event_type,
            source: None,
            target: None,
            player: None,
            other_entities: Vec::new(),
            values: Vec::new(),
            zones: Vec::new(),
            tags: Vec::new(),
        }
    }

    /// Set the source entity (builder pattern).
    #[must_use]
    pub fn with_source(mut self, source: EntityId) -> Self {
        self.source = Some(source);
        self
    }

    /// Set the target entity (builder pattern).
    #[must_use]
    pub fn with_target(mut self, target: EntityId) -> Self {
        self.target = Some(target);
        self
    }

    /// Set the associated player (builder pattern).
    #[must_use]
    pub fn with_player(mut self, player: PlayerId) -> Self {
        self.player = Some(player);
        self
    }

    /// Add another entity (builder pattern).
    #[must_use]
    pub fn with_entity(mut self, entity: EntityId) -> Self {
        self.other_entities.push(entity);
        self
    }

    /// Add a numeric value (builder pattern).
    #[must_use]
    pub fn with_value(mut self, value: i64) -> Self {
        self.values.push(value);
        self
    }

    /// Add zone information (builder pattern).
    #[must_use]
    pub fn with_zone(mut self, zone: ZoneId) -> Self {
        self.zones.push(zone);
        self
    }

    /// Add a tag (builder pattern).
    #[must_use]
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Get the first value, or a default.
    #[must_use]
    pub fn value(&self, index: usize, default: i64) -> i64 {
        self.values.get(index).copied().unwrap_or(default)
    }

    /// Get the first zone, or None.
    #[must_use]
    pub fn zone(&self, index: usize) -> Option<ZoneId> {
        self.zones.get(index).copied()
    }

    /// Check if event has a specific tag.
    #[must_use]
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }
}

/// Builder for common event patterns.
impl GameEvent {
    /// Create a damage event.
    ///
    /// Values[0] = damage amount
    pub fn damage(event_type: EventTypeId, source: EntityId, target: EntityId, amount: i64) -> Self {
        Self::new(event_type)
            .with_source(source)
            .with_target(target)
            .with_value(amount)
    }

    /// Create a zone change event.
    ///
    /// Zones[0] = from zone, Zones[1] = to zone
    pub fn zone_change(
        event_type: EventTypeId,
        card: EntityId,
        from: ZoneId,
        to: ZoneId,
    ) -> Self {
        Self::new(event_type)
            .with_target(card)
            .with_zone(from)
            .with_zone(to)
    }

    /// Create a player-centric event (like turn start).
    pub fn for_player(event_type: EventTypeId, player: PlayerId) -> Self {
        Self::new(event_type).with_player(player)
    }

    /// Create a card-centric event (like card played).
    pub fn for_card(event_type: EventTypeId, card: EntityId, controller: PlayerId) -> Self {
        Self::new(event_type)
            .with_source(card)
            .with_player(controller)
    }
}

/// Configuration for an event type.
///
/// Games provide this at startup to document event types.
#[derive(Clone, Debug)]
pub struct EventTypeConfig {
    /// Unique identifier for this event type.
    pub id: EventTypeId,

    /// Human-readable name (for debugging/display).
    pub name: String,

    /// Description of when this event fires.
    pub description: String,
}

impl EventTypeConfig {
    /// Create a new event type configuration.
    pub fn new(id: EventTypeId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            description: String::new(),
        }
    }

    /// Add a description (builder pattern).
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_type_id() {
        let id = EventTypeId::new(5);
        assert_eq!(id.raw(), 5);
        assert_eq!(format!("{}", id), "EventType(5)");
    }

    #[test]
    fn test_game_event_builder() {
        let event = GameEvent::new(EventTypeId::new(1))
            .with_source(EntityId(10))
            .with_target(EntityId(20))
            .with_player(PlayerId::new(0))
            .with_value(5)
            .with_zone(ZoneId::new(0))
            .with_tag("combat");

        assert_eq!(event.event_type, EventTypeId::new(1));
        assert_eq!(event.source, Some(EntityId(10)));
        assert_eq!(event.target, Some(EntityId(20)));
        assert_eq!(event.player, Some(PlayerId::new(0)));
        assert_eq!(event.value(0, 0), 5);
        assert_eq!(event.zone(0), Some(ZoneId::new(0)));
        assert!(event.has_tag("combat"));
        assert!(!event.has_tag("other"));
    }

    #[test]
    fn test_damage_event() {
        let event = GameEvent::damage(
            EventTypeId::new(1),
            EntityId(10),
            EntityId(20),
            3,
        );

        assert_eq!(event.source, Some(EntityId(10)));
        assert_eq!(event.target, Some(EntityId(20)));
        assert_eq!(event.value(0, 0), 3);
    }

    #[test]
    fn test_zone_change_event() {
        let event = GameEvent::zone_change(
            EventTypeId::new(2),
            EntityId(10),
            ZoneId::new(0),
            ZoneId::new(1),
        );

        assert_eq!(event.target, Some(EntityId(10)));
        assert_eq!(event.zone(0), Some(ZoneId::new(0)));
        assert_eq!(event.zone(1), Some(ZoneId::new(1)));
    }

    #[test]
    fn test_player_event() {
        let event = GameEvent::for_player(EventTypeId::new(3), PlayerId::new(2));

        assert_eq!(event.player, Some(PlayerId::new(2)));
    }

    #[test]
    fn test_card_event() {
        let event = GameEvent::for_card(EventTypeId::new(4), EntityId(15), PlayerId::new(1));

        assert_eq!(event.source, Some(EntityId(15)));
        assert_eq!(event.player, Some(PlayerId::new(1)));
    }

    #[test]
    fn test_event_config() {
        let config = EventTypeConfig::new(EventTypeId::new(1), "DamageDealt")
            .with_description("Fired when damage is dealt to an entity");

        assert_eq!(config.id, EventTypeId::new(1));
        assert_eq!(config.name, "DamageDealt");
        assert!(!config.description.is_empty());
    }

    #[test]
    fn test_event_serialization() {
        let event = GameEvent::damage(EventTypeId::new(1), EntityId(10), EntityId(20), 5);
        let json = serde_json::to_string(&event).unwrap();
        let deserialized: GameEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, deserialized);
    }
}

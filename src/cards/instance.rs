//! Card instances - runtime card state.
//!
//! `CardInstance` represents a specific card in a game at a specific moment.
//! It tracks mutable state like damage, counters, and current zone.
//!
//! ## Neutral Cards
//!
//! Cards can have no owner for games with shared cards:
//! - Deckbuilders with shared market
//! - Neutral obstacles
//! - Game-controlled entities

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use super::definition::CardId;
use crate::core::config::ZoneId;
use crate::core::entity::EntityId;
use crate::core::player::PlayerId;

/// A card instance in a game.
///
/// Tracks mutable state for a specific card during gameplay.
///
/// ## State Values (i64 only)
///
/// The `state` field uses `FxHashMap<String, i64>` for MCTS performance:
/// - Fast hashing (millions of state comparisons)
/// - Efficient cloning
///
/// To store non-integer values:
/// - Booleans: use 0/1
/// - Entity references: use EntityId.0 as i64
/// - Enums: use discriminant values
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CardInstance {
    /// Unique entity ID for this instance.
    pub entity_id: EntityId,

    /// Reference to the card definition.
    pub card_id: CardId,

    /// Owner (who started with this card). `None` for neutral cards.
    pub owner: Option<PlayerId>,

    /// Controller (who currently controls it). `None` for uncontrolled.
    pub controller: Option<PlayerId>,

    /// Current zone.
    pub zone: ZoneId,

    /// Is this card face-down?
    pub face_down: bool,

    /// Mutable instance state (damage, counters, tapped, etc.)
    #[serde(default)]
    pub state: FxHashMap<String, i64>,
}

impl std::hash::Hash for CardInstance {
    fn hash<H: std::hash::Hasher>(&self, hasher: &mut H) {
        self.entity_id.hash(hasher);
        self.card_id.hash(hasher);
        self.owner.hash(hasher);
        self.controller.hash(hasher);
        self.zone.hash(hasher);
        self.face_down.hash(hasher);

        // Hash state entries in sorted order for determinism
        let mut entries: Vec<_> = self.state.iter().collect();
        entries.sort_by_key(|(k, _)| *k);
        for (k, v) in entries {
            k.hash(hasher);
            v.hash(hasher);
        }
    }
}

impl CardInstance {
    /// Create a card instance with an owner.
    #[must_use]
    pub fn new(entity_id: EntityId, card_id: CardId, owner: PlayerId, zone: ZoneId) -> Self {
        Self {
            entity_id,
            card_id,
            owner: Some(owner),
            controller: Some(owner),
            zone,
            face_down: false,
            state: FxHashMap::default(),
        }
    }

    /// Create a neutral card instance (no owner/controller).
    ///
    /// Use for shared cards, obstacles, or game-controlled entities.
    #[must_use]
    pub fn neutral(entity_id: EntityId, card_id: CardId, zone: ZoneId) -> Self {
        Self {
            entity_id,
            card_id,
            owner: None,
            controller: None,
            zone,
            face_down: false,
            state: FxHashMap::default(),
        }
    }

    /// Check if this is a neutral (ownerless) card.
    #[must_use]
    pub fn is_neutral(&self) -> bool {
        self.owner.is_none()
    }

    /// Get the owner, panicking if neutral.
    ///
    /// Use when you know the card must have an owner.
    #[must_use]
    pub fn owner_unchecked(&self) -> PlayerId {
        self.owner.expect("Card has no owner")
    }

    /// Get the controller, panicking if uncontrolled.
    #[must_use]
    pub fn controller_unchecked(&self) -> PlayerId {
        self.controller.expect("Card has no controller")
    }

    /// Set the controller.
    pub fn set_controller(&mut self, controller: Option<PlayerId>) {
        self.controller = controller;
    }

    /// Get a state value with a default.
    #[must_use]
    pub fn get_state(&self, key: &str, default: i64) -> i64 {
        self.state.get(key).copied().unwrap_or(default)
    }

    /// Set a state value.
    pub fn set_state(&mut self, key: impl Into<String>, value: i64) {
        self.state.insert(key.into(), value);
    }

    /// Modify a state value by delta.
    pub fn modify_state(&mut self, key: &str, delta: i64) {
        let current = self.get_state(key, 0);
        self.state.insert(key.to_string(), current + delta);
    }

    /// Check if a state flag is set (non-zero).
    #[must_use]
    pub fn has_flag(&self, key: &str) -> bool {
        self.get_state(key, 0) != 0
    }

    /// Set a boolean flag (1 for true, 0 for false).
    pub fn set_flag(&mut self, key: impl Into<String>, value: bool) {
        self.set_state(key, if value { 1 } else { 0 });
    }

    /// Clear all state (e.g., when card changes zones).
    pub fn clear_state(&mut self) {
        self.state.clear();
    }

    /// Clear specific state keys.
    pub fn clear_state_keys(&mut self, keys: &[&str]) {
        for key in keys {
            self.state.remove(*key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_zone() -> ZoneId {
        ZoneId::new(0)
    }

    #[test]
    fn test_card_instance_new() {
        let instance = CardInstance::new(
            EntityId(10),
            CardId::new(1),
            PlayerId::new(0),
            test_zone(),
        );

        assert_eq!(instance.entity_id, EntityId(10));
        assert_eq!(instance.card_id, CardId::new(1));
        assert_eq!(instance.owner, Some(PlayerId::new(0)));
        assert_eq!(instance.controller, Some(PlayerId::new(0)));
        assert!(!instance.is_neutral());
    }

    #[test]
    fn test_card_instance_neutral() {
        let instance = CardInstance::neutral(EntityId(10), CardId::new(1), test_zone());

        assert!(instance.is_neutral());
        assert!(instance.owner.is_none());
        assert!(instance.controller.is_none());
    }

    #[test]
    fn test_card_instance_state() {
        let mut instance = CardInstance::new(
            EntityId(10),
            CardId::new(1),
            PlayerId::new(0),
            test_zone(),
        );

        assert_eq!(instance.get_state("damage", 0), 0);

        instance.set_state("damage", 3);
        assert_eq!(instance.get_state("damage", 0), 3);

        instance.modify_state("damage", 2);
        assert_eq!(instance.get_state("damage", 0), 5);
    }

    #[test]
    fn test_card_instance_flags() {
        let mut instance = CardInstance::new(
            EntityId(10),
            CardId::new(1),
            PlayerId::new(0),
            test_zone(),
        );

        assert!(!instance.has_flag("tapped"));

        instance.set_flag("tapped", true);
        assert!(instance.has_flag("tapped"));

        instance.set_flag("tapped", false);
        assert!(!instance.has_flag("tapped"));
    }

    #[test]
    fn test_card_instance_clear_state() {
        let mut instance = CardInstance::new(
            EntityId(10),
            CardId::new(1),
            PlayerId::new(0),
            test_zone(),
        );

        instance.set_state("damage", 3);
        instance.set_state("counters", 2);

        instance.clear_state();
        assert_eq!(instance.get_state("damage", 0), 0);
        assert_eq!(instance.get_state("counters", 0), 0);
    }

    #[test]
    fn test_card_instance_controller_change() {
        let mut instance = CardInstance::new(
            EntityId(10),
            CardId::new(1),
            PlayerId::new(0),
            test_zone(),
        );

        assert_eq!(instance.controller, Some(PlayerId::new(0)));

        instance.set_controller(Some(PlayerId::new(1)));
        assert_eq!(instance.controller, Some(PlayerId::new(1)));
        assert_eq!(instance.owner, Some(PlayerId::new(0))); // Owner unchanged
    }

    #[test]
    fn test_card_instance_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let instance1 = CardInstance::new(
            EntityId(10),
            CardId::new(1),
            PlayerId::new(0),
            test_zone(),
        );

        let instance2 = CardInstance::new(
            EntityId(10),
            CardId::new(1),
            PlayerId::new(0),
            test_zone(),
        );

        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        instance1.hash(&mut h1);
        instance2.hash(&mut h2);

        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn test_card_instance_serialization() {
        let mut instance = CardInstance::new(
            EntityId(10),
            CardId::new(1),
            PlayerId::new(0),
            test_zone(),
        );
        instance.set_state("damage", 3);

        let json = serde_json::to_string(&instance).unwrap();
        let deserialized: CardInstance = serde_json::from_str(&json).unwrap();

        assert_eq!(instance, deserialized);
    }
}

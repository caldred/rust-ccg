//! Effect definitions.
//!
//! Effects are atomic game actions that can be composed
//! into complex abilities. They operate on players, cards,
//! and game state without game-specific knowledge.

use serde::{Deserialize, Serialize};

use crate::core::{EntityId, PlayerId, ZoneId};
use crate::zones::ZonePosition;

/// An atomic game effect.
///
/// Effects are the fundamental building blocks of card abilities.
/// They're intentionally simple and game-agnostic - games give
/// meaning to state keys and zones through configuration.
///
/// ## Player State Effects
///
/// Modify values in `player_state` (life, mana, etc.):
/// - `ModifyPlayerState`: Add/subtract from a state value
/// - `SetPlayerState`: Set a state value directly
///
/// ## Card Movement Effects
///
/// Move cards between zones:
/// - `MoveCard`: Move a card to a new zone
/// - `DrawCards`: Draw from deck to hand
/// - `ShuffleZone`: Randomize a zone's order
///
/// ## Card State Effects
///
/// Modify card instance state:
/// - `ModifyCardState`: Add/subtract from a card state value
/// - `SetCardState`: Set a card state value directly
///
/// ## Composite Effects
///
/// - `Batch`: Execute multiple effects in sequence
/// - `Conditional`: Execute an effect if a condition is met
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Effect {
    // === Player State ===

    /// Modify a player state value (add delta).
    /// Target: player entity
    ModifyPlayerState {
        key: String,
        delta: i64,
    },

    /// Set a player state value directly.
    /// Target: player entity
    SetPlayerState {
        key: String,
        value: i64,
    },

    // === Card Movement ===

    /// Move a card to a zone.
    /// Target: card entity
    MoveCard {
        destination: ZoneId,
        position: Option<ZonePosition>,
    },

    /// Draw cards from a player's deck to hand.
    /// Target: player entity
    /// Requires game to define deck and hand zone conventions.
    DrawCards {
        count: usize,
        /// Zone to draw from (deck). If None, game must provide.
        from_zone: Option<ZoneId>,
        /// Zone to draw to (hand). If None, game must provide.
        to_zone: Option<ZoneId>,
    },

    /// Shuffle a zone's card order.
    /// Target: zone (not entity) - applied via resolver
    ShuffleZone {
        zone: ZoneId,
    },

    // === Card State ===

    /// Modify a card state value (add delta).
    /// Target: card entity
    ModifyCardState {
        key: String,
        delta: i64,
    },

    /// Set a card state value directly.
    /// Target: card entity
    SetCardState {
        key: String,
        value: i64,
    },

    // === Turn/Game State ===

    /// Modify turn state value (add delta).
    ModifyTurnState {
        key: String,
        delta: i64,
    },

    /// Set turn state value directly.
    SetTurnState {
        key: String,
        value: i64,
    },

    // === Composite ===

    /// Execute multiple effects in sequence.
    Batch(Vec<Effect>),

    /// Execute effect only if condition is met.
    /// The condition is evaluated by the resolver callback.
    Conditional {
        condition_key: String,
        effect: Box<Effect>,
    },
}

impl Effect {
    /// Create a damage effect (modify player's "life" state).
    pub fn damage(amount: i64) -> Self {
        Self::ModifyPlayerState {
            key: "life".to_string(),
            delta: -amount,
        }
    }

    /// Create a heal effect (modify player's "life" state).
    pub fn heal(amount: i64) -> Self {
        Self::ModifyPlayerState {
            key: "life".to_string(),
            delta: amount,
        }
    }

    /// Create a draw cards effect.
    pub fn draw(count: usize) -> Self {
        Self::DrawCards {
            count,
            from_zone: None,
            to_zone: None,
        }
    }

    /// Create a move card effect.
    pub fn move_to(zone: ZoneId) -> Self {
        Self::MoveCard {
            destination: zone,
            position: None,
        }
    }

    /// Create a move card to top of zone effect.
    pub fn move_to_top(zone: ZoneId) -> Self {
        Self::MoveCard {
            destination: zone,
            position: Some(ZonePosition::Top),
        }
    }

    /// Create a move card to bottom of zone effect.
    pub fn move_to_bottom(zone: ZoneId) -> Self {
        Self::MoveCard {
            destination: zone,
            position: Some(ZonePosition::Bottom),
        }
    }

    /// Create a modify player state effect.
    pub fn modify_player(key: impl Into<String>, delta: i64) -> Self {
        Self::ModifyPlayerState {
            key: key.into(),
            delta,
        }
    }

    /// Create a set player state effect.
    pub fn set_player(key: impl Into<String>, value: i64) -> Self {
        Self::SetPlayerState {
            key: key.into(),
            value,
        }
    }

    /// Create a modify card state effect.
    pub fn modify_card(key: impl Into<String>, delta: i64) -> Self {
        Self::ModifyCardState {
            key: key.into(),
            delta,
        }
    }

    /// Create a set card state effect.
    pub fn set_card(key: impl Into<String>, value: i64) -> Self {
        Self::SetCardState {
            key: key.into(),
            value,
        }
    }

    /// Create a batch of effects.
    pub fn batch(effects: impl IntoIterator<Item = Effect>) -> Self {
        Self::Batch(effects.into_iter().collect())
    }
}

/// A batch of targeted effects to resolve.
///
/// Pairs effects with their targets for resolution.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EffectBatch {
    entries: Vec<EffectEntry>,
}

/// A single entry in an effect batch.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EffectEntry {
    /// The effect to apply.
    pub effect: Effect,
    /// Targets for the effect.
    pub targets: Vec<EntityId>,
}

impl EffectBatch {
    /// Create an empty batch.
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Add an effect with targets.
    pub fn add(&mut self, effect: Effect, targets: impl IntoIterator<Item = EntityId>) {
        self.entries.push(EffectEntry {
            effect,
            targets: targets.into_iter().collect(),
        });
    }

    /// Add an effect targeting a single entity.
    pub fn add_single(&mut self, effect: Effect, target: EntityId) {
        self.add(effect, std::iter::once(target));
    }

    /// Add an effect targeting a player.
    pub fn add_player(&mut self, effect: Effect, player: PlayerId) {
        self.add(effect, std::iter::once(EntityId::player_id(player.0)));
    }

    /// Add a zone effect (no entity target).
    pub fn add_zone(&mut self, effect: Effect) {
        self.entries.push(EffectEntry {
            effect,
            targets: Vec::new(),
        });
    }

    /// Iterate over entries.
    pub fn iter(&self) -> impl Iterator<Item = &EffectEntry> {
        self.entries.iter()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

impl Default for EffectBatch {
    fn default() -> Self {
        Self::new()
    }
}

impl IntoIterator for EffectBatch {
    type Item = EffectEntry;
    type IntoIter = std::vec::IntoIter<EffectEntry>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_damage_effect() {
        let effect = Effect::damage(3);
        match effect {
            Effect::ModifyPlayerState { key, delta } => {
                assert_eq!(key, "life");
                assert_eq!(delta, -3);
            }
            _ => panic!("Expected ModifyPlayerState"),
        }
    }

    #[test]
    fn test_heal_effect() {
        let effect = Effect::heal(5);
        match effect {
            Effect::ModifyPlayerState { key, delta } => {
                assert_eq!(key, "life");
                assert_eq!(delta, 5);
            }
            _ => panic!("Expected ModifyPlayerState"),
        }
    }

    #[test]
    fn test_draw_effect() {
        let effect = Effect::draw(2);
        match effect {
            Effect::DrawCards { count, from_zone, to_zone } => {
                assert_eq!(count, 2);
                assert!(from_zone.is_none());
                assert!(to_zone.is_none());
            }
            _ => panic!("Expected DrawCards"),
        }
    }

    #[test]
    fn test_move_effect() {
        let zone = ZoneId::new(5);
        let effect = Effect::move_to_top(zone);
        match effect {
            Effect::MoveCard { destination, position } => {
                assert_eq!(destination, zone);
                assert_eq!(position, Some(ZonePosition::Top));
            }
            _ => panic!("Expected MoveCard"),
        }
    }

    #[test]
    fn test_batch_effect() {
        let effect = Effect::batch([
            Effect::damage(3),
            Effect::draw(1),
        ]);

        match effect {
            Effect::Batch(effects) => {
                assert_eq!(effects.len(), 2);
            }
            _ => panic!("Expected Batch"),
        }
    }

    #[test]
    fn test_effect_batch() {
        let mut batch = EffectBatch::new();

        batch.add_player(Effect::damage(3), PlayerId::new(1));
        batch.add_single(Effect::modify_card("damage", 2), EntityId(10));

        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());

        let entries: Vec<_> = batch.iter().collect();
        assert_eq!(entries[0].targets, vec![EntityId(1)]);
        assert_eq!(entries[1].targets, vec![EntityId(10)]);
    }

    #[test]
    fn test_effect_serialization() {
        let effect = Effect::damage(5);
        let json = serde_json::to_string(&effect).unwrap();
        let deserialized: Effect = serde_json::from_str(&json).unwrap();
        assert_eq!(effect, deserialized);
    }
}

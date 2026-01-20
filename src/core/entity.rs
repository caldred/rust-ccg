//! Entity identification system.
//!
//! Every game object (player, card, token, ability) has a unique `EntityId`.
//!
//! ## ID Layout
//!
//! IDs are allocated as follows:
//! - `0..player_count`: Reserved for players
//! - `player_count..`: Cards and other entities
//!
//! This layout is **configured per-game** via `player_count`, not hardcoded.
//!
//! ## Usage
//!
//! ```
//! use rust_ccg::core::EntityId;
//!
//! let player_count = 4;
//!
//! // Player entity IDs
//! let player_0 = EntityId::player_id(0);
//! let player_3 = EntityId::player_id(3);
//!
//! assert!(player_0.is_player(player_count));
//! assert!(player_3.is_player(player_count));
//!
//! // Card entity ID (allocated by game)
//! let card = EntityId(10);
//! assert!(!card.is_player(player_count));
//! ```

use serde::{Deserialize, Serialize};

/// Unique identifier for any game entity.
///
/// Players, cards, tokens, and abilities all have EntityIds.
/// Use `is_player(player_count)` to check if an ID refers to a player.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntityId(pub u32);

impl EntityId {
    /// Create an entity ID for a player by index.
    ///
    /// Player indices are 0-based: player 0, player 1, etc.
    #[must_use]
    pub const fn player_id(index: u8) -> Self {
        Self(index as u32)
    }

    /// Get the first entity ID available for non-player entities.
    ///
    /// In a game with `player_count` players, entity IDs 0..player_count
    /// are reserved for players. This returns `player_count` as the first
    /// available ID for cards, tokens, etc.
    #[must_use]
    pub const fn first_non_player(player_count: usize) -> u32 {
        player_count as u32
    }

    /// Check if this entity ID refers to a player.
    ///
    /// Player IDs are 0..player_count. This requires knowing the player count.
    ///
    /// ```
    /// use rust_ccg::core::EntityId;
    ///
    /// let id = EntityId(1);
    /// assert!(id.is_player(2));  // Valid player in 2-player game
    /// assert!(id.is_player(4));  // Valid player in 4-player game
    /// assert!(!id.is_player(1)); // NOT valid in 1-player game (only player 0)
    /// ```
    #[must_use]
    pub const fn is_player(self, player_count: usize) -> bool {
        self.0 < player_count as u32
    }

    /// Convert to player index if this is a player entity.
    ///
    /// Returns `Some(index)` if this is a valid player ID, `None` otherwise.
    ///
    /// ```
    /// use rust_ccg::core::EntityId;
    ///
    /// let player = EntityId(2);
    /// assert_eq!(player.as_player_index(4), Some(2)); // Player 2 in 4-player
    /// assert_eq!(player.as_player_index(2), None);    // Not a player in 2-player
    ///
    /// let card = EntityId(10);
    /// assert_eq!(card.as_player_index(4), None);      // Cards are never players
    /// ```
    #[must_use]
    pub const fn as_player_index(self, player_count: usize) -> Option<u8> {
        if self.is_player(player_count) {
            Some(self.0 as u8)
        } else {
            None
        }
    }

    /// Get the raw ID value.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Convert to PlayerId if this is a player entity.
    ///
    /// Returns `Some(PlayerId)` if this is a valid player ID, `None` otherwise.
    /// This is a convenience wrapper around `as_player_index`.
    ///
    /// ```
    /// use rust_ccg::core::{EntityId, PlayerId};
    ///
    /// let player = EntityId(2);
    /// assert_eq!(player.as_player(4), Some(PlayerId::new(2)));
    /// assert_eq!(player.as_player(2), None); // Not a player in 2-player game
    /// ```
    #[must_use]
    pub fn as_player(self, player_count: usize) -> Option<super::PlayerId> {
        self.as_player_index(player_count).map(super::PlayerId::new)
    }

    /// Create entity ID for a player.
    ///
    /// Convenience method that takes a PlayerId instead of a raw index.
    ///
    /// ```
    /// use rust_ccg::core::{EntityId, PlayerId};
    ///
    /// let entity = EntityId::player(PlayerId::new(2));
    /// assert_eq!(entity.0, 2);
    /// ```
    #[must_use]
    pub const fn player(id: super::PlayerId) -> Self {
        Self(id.0 as u32)
    }
}

impl From<u32> for EntityId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl std::fmt::Display for EntityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Entity({})", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_player_ids() {
        let p0 = EntityId::player_id(0);
        let p1 = EntityId::player_id(1);
        let p2 = EntityId::player_id(2);
        let p3 = EntityId::player_id(3);

        assert_eq!(p0.0, 0);
        assert_eq!(p1.0, 1);
        assert_eq!(p2.0, 2);
        assert_eq!(p3.0, 3);
    }

    #[test]
    fn test_is_player_2_player() {
        let player_count = 2;

        assert!(EntityId(0).is_player(player_count));
        assert!(EntityId(1).is_player(player_count));
        assert!(!EntityId(2).is_player(player_count));
        assert!(!EntityId(100).is_player(player_count));
    }

    #[test]
    fn test_is_player_4_player() {
        let player_count = 4;

        assert!(EntityId(0).is_player(player_count));
        assert!(EntityId(1).is_player(player_count));
        assert!(EntityId(2).is_player(player_count));
        assert!(EntityId(3).is_player(player_count));
        assert!(!EntityId(4).is_player(player_count));
        assert!(!EntityId(100).is_player(player_count));
    }

    #[test]
    fn test_is_player_1_player() {
        let player_count = 1;

        assert!(EntityId(0).is_player(player_count));
        assert!(!EntityId(1).is_player(player_count));
    }

    #[test]
    fn test_as_player_index() {
        assert_eq!(EntityId(0).as_player_index(4), Some(0));
        assert_eq!(EntityId(3).as_player_index(4), Some(3));
        assert_eq!(EntityId(4).as_player_index(4), None);
        assert_eq!(EntityId(2).as_player_index(2), None);
    }

    #[test]
    fn test_first_non_player() {
        assert_eq!(EntityId::first_non_player(2), 2);
        assert_eq!(EntityId::first_non_player(4), 4);
        assert_eq!(EntityId::first_non_player(8), 8);
    }

    #[test]
    fn test_as_player() {
        use super::super::PlayerId;

        assert_eq!(EntityId(0).as_player(4), Some(PlayerId::new(0)));
        assert_eq!(EntityId(3).as_player(4), Some(PlayerId::new(3)));
        assert_eq!(EntityId(4).as_player(4), None);
        assert_eq!(EntityId(2).as_player(2), None);
    }

    #[test]
    fn test_player_from_player_id() {
        use super::super::PlayerId;

        let entity = EntityId::player(PlayerId::new(3));
        assert_eq!(entity.0, 3);
        assert!(entity.is_player(4));
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", EntityId(42)), "Entity(42)");
    }

    #[test]
    fn test_serialization() {
        let id = EntityId(123);
        let json = serde_json::to_string(&id).unwrap();
        let deserialized: EntityId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, deserialized);
    }
}

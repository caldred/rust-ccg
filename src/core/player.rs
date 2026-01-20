//! Player identification and per-player data storage.
//!
//! ## PlayerId
//!
//! Type-safe player identifier supporting 1-255 players.
//!
//! ## PlayerMap
//!
//! Efficient per-player data storage backed by `Vec` for O(1) access.
//! Supports iteration and indexing by `PlayerId`.

use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};

/// Player identifier supporting 1-255 players.
///
/// Player indices are 0-based: the first player is `PlayerId(0)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PlayerId(pub u8);

impl PlayerId {
    /// Create a new player ID.
    #[must_use]
    pub const fn new(id: u8) -> Self {
        Self(id)
    }

    /// Get the raw player index (0-based).
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// Iterate over all player IDs for a game with `player_count` players.
    ///
    /// ```
    /// use rust_ccg::core::PlayerId;
    ///
    /// let players: Vec<_> = PlayerId::all(4).collect();
    /// assert_eq!(players.len(), 4);
    /// assert_eq!(players[0], PlayerId::new(0));
    /// assert_eq!(players[3], PlayerId::new(3));
    /// ```
    pub fn all(player_count: usize) -> impl Iterator<Item = PlayerId> {
        (0..player_count as u8).map(PlayerId)
    }
}

impl std::fmt::Display for PlayerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Player {}", self.0)
    }
}

/// Per-player data storage with O(1) access.
///
/// Backed by a `Vec<T>` with one entry per player.
/// Use `PlayerMap::new()` to create with a factory function,
/// or `PlayerMap::with_value()` to initialize all entries to the same value.
///
/// ## Example
///
/// ```
/// use rust_ccg::core::{PlayerId, PlayerMap};
///
/// // Create with factory
/// let mut life: PlayerMap<i32> = PlayerMap::new(4, |_| 20);
///
/// // Access by player
/// assert_eq!(life[PlayerId::new(0)], 20);
///
/// // Modify
/// life[PlayerId::new(1)] = 15;
/// assert_eq!(life[PlayerId::new(1)], 15);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PlayerMap<T> {
    data: Vec<T>,
}

impl<T> PlayerMap<T> {
    /// Create a new PlayerMap with values from a factory function.
    ///
    /// The factory receives the `PlayerId` for each player.
    pub fn new(player_count: usize, factory: impl Fn(PlayerId) -> T) -> Self {
        assert!(player_count > 0, "Must have at least 1 player");
        assert!(player_count <= 255, "At most 255 players supported");

        let data = (0..player_count as u8)
            .map(|i| factory(PlayerId(i)))
            .collect();

        Self { data }
    }

    /// Create a new PlayerMap with all entries set to the same value.
    pub fn with_value(player_count: usize, value: T) -> Self
    where
        T: Clone,
    {
        Self::new(player_count, |_| value.clone())
    }

    /// Create a new PlayerMap with default values.
    pub fn with_default(player_count: usize) -> Self
    where
        T: Default,
    {
        Self::new(player_count, |_| T::default())
    }

    /// Get the number of players.
    #[must_use]
    pub fn player_count(&self) -> usize {
        self.data.len()
    }

    /// Get a reference to a player's data.
    #[must_use]
    pub fn get(&self, player: PlayerId) -> &T {
        &self.data[player.index()]
    }

    /// Get a mutable reference to a player's data.
    pub fn get_mut(&mut self, player: PlayerId) -> &mut T {
        &mut self.data[player.index()]
    }

    /// Iterate over (PlayerId, &T) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (PlayerId, &T)> {
        self.data
            .iter()
            .enumerate()
            .map(|(i, v)| (PlayerId(i as u8), v))
    }

    /// Iterate over (PlayerId, &mut T) pairs.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (PlayerId, &mut T)> {
        self.data
            .iter_mut()
            .enumerate()
            .map(|(i, v)| (PlayerId(i as u8), v))
    }

    /// Iterate over all player IDs.
    pub fn player_ids(&self) -> impl Iterator<Item = PlayerId> {
        (0..self.data.len() as u8).map(PlayerId)
    }
}

impl<T> Index<PlayerId> for PlayerMap<T> {
    type Output = T;

    fn index(&self, player: PlayerId) -> &Self::Output {
        self.get(player)
    }
}

impl<T> IndexMut<PlayerId> for PlayerMap<T> {
    fn index_mut(&mut self, player: PlayerId) -> &mut Self::Output {
        self.get_mut(player)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_player_id_basics() {
        let p0 = PlayerId::new(0);
        let p1 = PlayerId::new(1);

        assert_eq!(p0.index(), 0);
        assert_eq!(p1.index(), 1);
        assert_eq!(format!("{}", p0), "Player 0");
    }

    #[test]
    fn test_player_id_all() {
        let players: Vec<_> = PlayerId::all(4).collect();
        assert_eq!(players.len(), 4);
        assert_eq!(players[0], PlayerId::new(0));
        assert_eq!(players[1], PlayerId::new(1));
        assert_eq!(players[2], PlayerId::new(2));
        assert_eq!(players[3], PlayerId::new(3));
    }

    #[test]
    fn test_player_map_new() {
        let map: PlayerMap<i32> = PlayerMap::new(4, |p| p.index() as i32 * 10);

        assert_eq!(map[PlayerId::new(0)], 0);
        assert_eq!(map[PlayerId::new(1)], 10);
        assert_eq!(map[PlayerId::new(2)], 20);
        assert_eq!(map[PlayerId::new(3)], 30);
    }

    #[test]
    fn test_player_map_with_value() {
        let map: PlayerMap<i32> = PlayerMap::with_value(3, 20);

        assert_eq!(map[PlayerId::new(0)], 20);
        assert_eq!(map[PlayerId::new(1)], 20);
        assert_eq!(map[PlayerId::new(2)], 20);
    }

    #[test]
    fn test_player_map_with_default() {
        let map: PlayerMap<Vec<i32>> = PlayerMap::with_default(2);

        assert!(map[PlayerId::new(0)].is_empty());
        assert!(map[PlayerId::new(1)].is_empty());
    }

    #[test]
    fn test_player_map_mutation() {
        let mut map: PlayerMap<i32> = PlayerMap::with_value(2, 0);

        map[PlayerId::new(0)] = 10;
        map[PlayerId::new(1)] = 20;

        assert_eq!(map[PlayerId::new(0)], 10);
        assert_eq!(map[PlayerId::new(1)], 20);
    }

    #[test]
    fn test_player_map_iter() {
        let map: PlayerMap<i32> = PlayerMap::new(3, |p| p.index() as i32);

        let pairs: Vec<_> = map.iter().collect();
        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[0], (PlayerId::new(0), &0));
        assert_eq!(pairs[1], (PlayerId::new(1), &1));
        assert_eq!(pairs[2], (PlayerId::new(2), &2));
    }

    #[test]
    fn test_player_map_player_count() {
        let map: PlayerMap<i32> = PlayerMap::with_value(5, 0);
        assert_eq!(map.player_count(), 5);
    }

    #[test]
    fn test_player_map_serialization() {
        let map: PlayerMap<i32> = PlayerMap::new(2, |p| p.index() as i32 + 1);
        let json = serde_json::to_string(&map).unwrap();
        let deserialized: PlayerMap<i32> = serde_json::from_str(&json).unwrap();
        assert_eq!(map, deserialized);
    }

    #[test]
    #[should_panic(expected = "Must have at least 1 player")]
    fn test_player_map_zero_players() {
        let _: PlayerMap<i32> = PlayerMap::with_value(0, 0);
    }
}

//! Zone manager for card locations and movement.
//!
//! The `ZoneManager` tracks where cards are located and handles movement
//! between zones. It supports:
//! - Ordered zones (library, stack) with explicit position control
//! - Unordered zones (battlefield) with set-like semantics
//! - Card lookup by entity ID
//! - Zone iteration

use rustc_hash::FxHashMap;

use crate::core::config::ZoneId;
use crate::core::entity::EntityId;
use crate::core::rng::GameRng;

use serde::{Deserialize, Serialize};

/// Position for inserting a card into an ordered zone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZonePosition {
    /// Add to top of zone (e.g., top of library).
    Top,
    /// Add to bottom of zone.
    Bottom,
    /// Insert at specific index (0 = bottom for libraries).
    Index(usize),
}

/// Manages card locations across zones.
///
/// Supports both ordered zones (where card order matters, like library)
/// and unordered zones (like battlefield).
///
/// ## Usage
///
/// ```
/// use rust_ccg::zones::{ZoneManager, ZonePosition};
/// use rust_ccg::core::{ZoneId, ZoneConfig, EntityId};
///
/// let mut manager = ZoneManager::new();
///
/// // Initialize an ordered zone (library)
/// let library = ZoneId::new(0);
/// manager.init_ordered_zone(library);
///
/// // Add cards to specific positions
/// manager.add_to_zone(EntityId(10), library, Some(ZonePosition::Top));
/// manager.add_to_zone(EntityId(11), library, Some(ZonePosition::Bottom));
///
/// // Get cards in order
/// let cards = manager.cards_in_zone_ordered(library);
/// ```
#[derive(Clone, Debug, Default)]
pub struct ZoneManager {
    /// Card locations: entity_id -> zone_id
    locations: FxHashMap<EntityId, ZoneId>,

    /// Ordered card lists for zones where order matters.
    /// Only populated for zones that call `init_ordered_zone`.
    zone_order: FxHashMap<ZoneId, Vec<EntityId>>,
}

impl ZoneManager {
    /// Create a new empty zone manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Initialize an ordered zone (like library or stack).
    ///
    /// Must be called before adding cards to ordered zones.
    pub fn init_ordered_zone(&mut self, zone: ZoneId) {
        self.zone_order.entry(zone).or_default();
    }

    /// Check if a zone is ordered.
    #[must_use]
    pub fn is_ordered(&self, zone: ZoneId) -> bool {
        self.zone_order.contains_key(&zone)
    }

    /// Add a card to a zone.
    ///
    /// For ordered zones, use `position` to control placement.
    /// For unordered zones, `position` is ignored.
    ///
    /// Panics if the entity is already in the manager.
    pub fn add_to_zone(&mut self, entity: EntityId, zone: ZoneId, position: Option<ZonePosition>) {
        if self.locations.contains_key(&entity) {
            panic!("Entity {:?} already exists in zone manager", entity);
        }

        self.locations.insert(entity, zone);

        if let Some(order) = self.zone_order.get_mut(&zone) {
            let pos = position.unwrap_or(ZonePosition::Top);
            match pos {
                ZonePosition::Top => order.push(entity),
                ZonePosition::Bottom => order.insert(0, entity),
                ZonePosition::Index(i) => {
                    let idx = i.min(order.len());
                    order.insert(idx, entity);
                }
            }
        }
    }

    /// Move a card from one zone to another.
    ///
    /// Returns the old zone, or `None` if the card wasn't found.
    pub fn move_to_zone(
        &mut self,
        entity: EntityId,
        new_zone: ZoneId,
        position: Option<ZonePosition>,
    ) -> Option<ZoneId> {
        let old_zone = self.locations.get(&entity).copied()?;

        if old_zone == new_zone {
            return Some(old_zone);
        }

        // Remove from old zone ordering
        if let Some(order) = self.zone_order.get_mut(&old_zone) {
            order.retain(|&e| e != entity);
        }

        // Update location
        self.locations.insert(entity, new_zone);

        // Add to new zone ordering
        if let Some(order) = self.zone_order.get_mut(&new_zone) {
            let pos = position.unwrap_or(ZonePosition::Top);
            match pos {
                ZonePosition::Top => order.push(entity),
                ZonePosition::Bottom => order.insert(0, entity),
                ZonePosition::Index(i) => {
                    let idx = i.min(order.len());
                    order.insert(idx, entity);
                }
            }
        }

        Some(old_zone)
    }

    /// Remove a card from the manager entirely.
    ///
    /// Returns the zone it was in, or `None` if not found.
    pub fn remove(&mut self, entity: EntityId) -> Option<ZoneId> {
        let zone = self.locations.remove(&entity)?;

        if let Some(order) = self.zone_order.get_mut(&zone) {
            order.retain(|&e| e != entity);
        }

        Some(zone)
    }

    /// Get the zone a card is in.
    #[must_use]
    pub fn get_zone(&self, entity: EntityId) -> Option<ZoneId> {
        self.locations.get(&entity).copied()
    }

    /// Check if a card is in a specific zone.
    #[must_use]
    pub fn is_in_zone(&self, entity: EntityId, zone: ZoneId) -> bool {
        self.locations.get(&entity) == Some(&zone)
    }

    /// Get all cards in a zone (unordered).
    pub fn cards_in_zone(&self, zone: ZoneId) -> impl Iterator<Item = EntityId> + '_ {
        self.locations
            .iter()
            .filter(move |(_, &z)| z == zone)
            .map(|(&e, _)| e)
    }

    /// Get cards in an ordered zone, in order.
    ///
    /// For libraries, index 0 is bottom, last index is top.
    /// Returns empty if zone is not ordered.
    #[must_use]
    pub fn cards_in_zone_ordered(&self, zone: ZoneId) -> &[EntityId] {
        self.zone_order.get(&zone).map_or(&[], |v| v.as_slice())
    }

    /// Get the number of cards in a zone.
    #[must_use]
    pub fn zone_size(&self, zone: ZoneId) -> usize {
        if let Some(order) = self.zone_order.get(&zone) {
            order.len()
        } else {
            self.cards_in_zone(zone).count()
        }
    }

    /// Get the top card of an ordered zone (last in the vec).
    #[must_use]
    pub fn top_card(&self, zone: ZoneId) -> Option<EntityId> {
        self.zone_order.get(&zone)?.last().copied()
    }

    /// Get the bottom card of an ordered zone (first in the vec).
    #[must_use]
    pub fn bottom_card(&self, zone: ZoneId) -> Option<EntityId> {
        self.zone_order.get(&zone)?.first().copied()
    }

    /// Remove and return the top card of an ordered zone.
    pub fn pop_top(&mut self, zone: ZoneId) -> Option<EntityId> {
        let order = self.zone_order.get_mut(&zone)?;
        let entity = order.pop()?;
        self.locations.remove(&entity);
        Some(entity)
    }

    /// Remove and return the bottom card of an ordered zone.
    pub fn pop_bottom(&mut self, zone: ZoneId) -> Option<EntityId> {
        let order = self.zone_order.get_mut(&zone)?;
        if order.is_empty() {
            return None;
        }
        let entity = order.remove(0);
        self.locations.remove(&entity);
        Some(entity)
    }

    /// Shuffle an ordered zone.
    pub fn shuffle_zone(&mut self, zone: ZoneId, rng: &mut GameRng) {
        if let Some(order) = self.zone_order.get_mut(&zone) {
            rng.shuffle(order);
        }
    }

    /// Get total number of cards tracked.
    #[must_use]
    pub fn total_cards(&self) -> usize {
        self.locations.len()
    }

    /// Check if the manager contains an entity.
    #[must_use]
    pub fn contains(&self, entity: EntityId) -> bool {
        self.locations.contains_key(&entity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_get() {
        let mut manager = ZoneManager::new();
        let zone = ZoneId::new(0);

        manager.add_to_zone(EntityId(10), zone, None);
        manager.add_to_zone(EntityId(11), zone, None);

        assert_eq!(manager.get_zone(EntityId(10)), Some(zone));
        assert_eq!(manager.get_zone(EntityId(11)), Some(zone));
        assert_eq!(manager.get_zone(EntityId(99)), None);
        assert!(manager.is_in_zone(EntityId(10), zone));
    }

    #[test]
    fn test_ordered_zone() {
        let mut manager = ZoneManager::new();
        let library = ZoneId::new(0);

        manager.init_ordered_zone(library);

        // Add cards: 10 to top, 11 to bottom, 12 to top
        manager.add_to_zone(EntityId(10), library, Some(ZonePosition::Top));
        manager.add_to_zone(EntityId(11), library, Some(ZonePosition::Bottom));
        manager.add_to_zone(EntityId(12), library, Some(ZonePosition::Top));

        // Order should be: [11, 10, 12] (bottom to top)
        let order = manager.cards_in_zone_ordered(library);
        assert_eq!(order, &[EntityId(11), EntityId(10), EntityId(12)]);

        assert_eq!(manager.top_card(library), Some(EntityId(12)));
        assert_eq!(manager.bottom_card(library), Some(EntityId(11)));
    }

    #[test]
    fn test_move_between_zones() {
        let mut manager = ZoneManager::new();
        let zone_a = ZoneId::new(0);
        let zone_b = ZoneId::new(1);

        manager.init_ordered_zone(zone_a);
        manager.init_ordered_zone(zone_b);

        manager.add_to_zone(EntityId(10), zone_a, Some(ZonePosition::Top));

        let old = manager.move_to_zone(EntityId(10), zone_b, Some(ZonePosition::Top));

        assert_eq!(old, Some(zone_a));
        assert_eq!(manager.get_zone(EntityId(10)), Some(zone_b));
        assert_eq!(manager.zone_size(zone_a), 0);
        assert_eq!(manager.zone_size(zone_b), 1);
    }

    #[test]
    fn test_remove() {
        let mut manager = ZoneManager::new();
        let zone = ZoneId::new(0);

        manager.init_ordered_zone(zone);
        manager.add_to_zone(EntityId(10), zone, Some(ZonePosition::Top));

        let removed = manager.remove(EntityId(10));
        assert_eq!(removed, Some(zone));
        assert!(!manager.contains(EntityId(10)));
        assert_eq!(manager.zone_size(zone), 0);
    }

    #[test]
    fn test_pop_top() {
        let mut manager = ZoneManager::new();
        let library = ZoneId::new(0);

        manager.init_ordered_zone(library);
        manager.add_to_zone(EntityId(10), library, Some(ZonePosition::Top));
        manager.add_to_zone(EntityId(11), library, Some(ZonePosition::Top));

        // Pop should return top (11)
        let popped = manager.pop_top(library);
        assert_eq!(popped, Some(EntityId(11)));
        assert!(!manager.contains(EntityId(11)));

        let popped = manager.pop_top(library);
        assert_eq!(popped, Some(EntityId(10)));

        let popped = manager.pop_top(library);
        assert_eq!(popped, None);
    }

    #[test]
    fn test_shuffle() {
        let mut manager = ZoneManager::new();
        let library = ZoneId::new(0);

        manager.init_ordered_zone(library);
        for i in 0..20 {
            manager.add_to_zone(EntityId(i), library, Some(ZonePosition::Top));
        }

        let before: Vec<_> = manager.cards_in_zone_ordered(library).to_vec();

        let mut rng = GameRng::new(42);
        manager.shuffle_zone(library, &mut rng);

        let after: Vec<_> = manager.cards_in_zone_ordered(library).to_vec();

        // Should be same elements, different order (very likely)
        assert_eq!(before.len(), after.len());
        assert_ne!(before, after);
    }

    #[test]
    fn test_unordered_zone() {
        let mut manager = ZoneManager::new();
        let battlefield = ZoneId::new(0);
        // Don't init_ordered_zone - it's unordered

        manager.add_to_zone(EntityId(10), battlefield, None);
        manager.add_to_zone(EntityId(11), battlefield, None);

        assert!(!manager.is_ordered(battlefield));
        assert_eq!(manager.zone_size(battlefield), 2);

        // cards_in_zone works for unordered
        let cards: Vec<_> = manager.cards_in_zone(battlefield).collect();
        assert!(cards.contains(&EntityId(10)));
        assert!(cards.contains(&EntityId(11)));

        // cards_in_zone_ordered returns empty for unordered
        assert!(manager.cards_in_zone_ordered(battlefield).is_empty());
    }

    #[test]
    fn test_position_index() {
        let mut manager = ZoneManager::new();
        let zone = ZoneId::new(0);

        manager.init_ordered_zone(zone);
        manager.add_to_zone(EntityId(10), zone, Some(ZonePosition::Top));
        manager.add_to_zone(EntityId(11), zone, Some(ZonePosition::Top));
        // Order: [10, 11]

        // Insert at index 1 (between 10 and 11)
        manager.add_to_zone(EntityId(12), zone, Some(ZonePosition::Index(1)));

        let order = manager.cards_in_zone_ordered(zone);
        assert_eq!(order, &[EntityId(10), EntityId(12), EntityId(11)]);
    }

    #[test]
    #[should_panic(expected = "Entity")]
    fn test_duplicate_entity_panics() {
        let mut manager = ZoneManager::new();
        let zone = ZoneId::new(0);

        manager.add_to_zone(EntityId(10), zone, None);
        manager.add_to_zone(EntityId(10), zone, None); // Should panic
    }

    #[test]
    fn test_total_cards() {
        let mut manager = ZoneManager::new();

        assert_eq!(manager.total_cards(), 0);

        manager.add_to_zone(EntityId(10), ZoneId::new(0), None);
        manager.add_to_zone(EntityId(11), ZoneId::new(1), None);

        assert_eq!(manager.total_cards(), 2);
    }
}

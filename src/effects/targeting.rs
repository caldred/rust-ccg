//! Effect targeting system.
//!
//! Defines how effects select their targets:
//! - `TargetSpec`: Specification for what can be targeted
//! - `TargetFilter`: Filters for valid targets
//! - `TargetSelector`: Algorithms for selecting targets

use serde::{Deserialize, Serialize};

use crate::cards::CardTypeId;
use crate::core::{EntityId, GameState, PlayerId, ZoneId};

/// Specification for effect targeting.
///
/// Describes what kind of entities can be targeted and how many.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetSpec {
    /// What kind of entities can be targeted.
    pub target_type: TargetType,
    /// Filters to apply to potential targets.
    pub filters: Vec<TargetFilter>,
    /// How many targets to select.
    pub count: TargetCount,
    /// Whether targeting is optional.
    pub optional: bool,
}

/// The type of entity that can be targeted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetType {
    /// Target players.
    Player,
    /// Target cards in specific zones.
    Card { zones: Vec<ZoneId> },
    /// Target any entity (player or card).
    Any,
}

/// Number of targets to select.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetCount {
    /// Exactly N targets.
    Exactly(usize),
    /// Up to N targets.
    UpTo(usize),
    /// At least N targets.
    AtLeast(usize),
    /// Between min and max targets.
    Range { min: usize, max: usize },
    /// All valid targets.
    All,
}

/// Filters for valid targets.
///
/// Game-agnostic filters that work across different card games.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetFilter {
    // === Player Filters ===

    /// Target must be an opponent of the acting player.
    Opponent,
    /// Target must be the acting player.
    Self_,
    /// Target must not be the acting player.
    NotSelf,
    /// Target must be a specific player.
    SpecificPlayer(PlayerId),

    // === Card Filters ===

    /// Target card must be in specified zone.
    InZone(ZoneId),
    /// Target card must have specified card type.
    ///
    /// **Note**: This filter requires CardRegistry access to work properly.
    /// Since TargetSelector doesn't have registry access, this filter
    /// always returns `false` (no match). Games should use `Custom` filters
    /// with registry access for card type filtering, or provide their own
    /// selector implementation.
    HasCardType(CardTypeId),
    /// Target card must be owned by specified player.
    OwnedBy(PlayerId),
    /// Target card must be controlled by specified player.
    ControlledBy(PlayerId),
    /// Target card must have state key with value in range.
    StateInRange {
        key: String,
        min: Option<i64>,
        max: Option<i64>,
    },

    // === Generic Filters ===

    /// Target must not be the source entity.
    NotSource,
    /// Custom filter (evaluated by game-specific code).
    Custom(String),
}

impl TargetSpec {
    /// Create a spec for targeting a single player.
    pub fn single_player() -> Self {
        Self {
            target_type: TargetType::Player,
            filters: Vec::new(),
            count: TargetCount::Exactly(1),
            optional: false,
        }
    }

    /// Create a spec for targeting a single opponent.
    pub fn single_opponent() -> Self {
        Self {
            target_type: TargetType::Player,
            filters: vec![TargetFilter::Opponent],
            count: TargetCount::Exactly(1),
            optional: false,
        }
    }

    /// Create a spec for targeting a single card in specified zones.
    pub fn single_card(zones: impl IntoIterator<Item = ZoneId>) -> Self {
        Self {
            target_type: TargetType::Card {
                zones: zones.into_iter().collect(),
            },
            filters: Vec::new(),
            count: TargetCount::Exactly(1),
            optional: false,
        }
    }

    /// Create a spec for targeting multiple cards.
    pub fn multiple_cards(zones: impl IntoIterator<Item = ZoneId>, count: TargetCount) -> Self {
        Self {
            target_type: TargetType::Card {
                zones: zones.into_iter().collect(),
            },
            filters: Vec::new(),
            count,
            optional: false,
        }
    }

    /// Add a filter (builder pattern).
    pub fn with_filter(mut self, filter: TargetFilter) -> Self {
        self.filters.push(filter);
        self
    }

    /// Make targeting optional (builder pattern).
    pub fn optional(mut self) -> Self {
        self.optional = true;
        self
    }
}

/// Selector for choosing targets based on a spec.
#[derive(Clone, Debug)]
pub struct TargetSelector {
    spec: TargetSpec,
    acting_player: PlayerId,
    source_entity: Option<EntityId>,
}

impl TargetSelector {
    /// Create a new target selector.
    pub fn new(spec: TargetSpec, acting_player: PlayerId) -> Self {
        Self {
            spec,
            acting_player,
            source_entity: None,
        }
    }

    /// Set the source entity (for NotSource filter).
    pub fn with_source(mut self, source: EntityId) -> Self {
        self.source_entity = Some(source);
        self
    }

    /// Get all valid targets from game state.
    pub fn valid_targets(&self, state: &GameState) -> Vec<EntityId> {
        let player_count = state.player_count();
        let mut targets = Vec::new();

        match &self.spec.target_type {
            TargetType::Player => {
                for player in PlayerId::all(player_count) {
                    let entity = EntityId::player(player);
                    if self.passes_filters(state, entity, player_count) {
                        targets.push(entity);
                    }
                }
            }
            TargetType::Card { zones } => {
                for zone in zones {
                    for entity in state.zones.cards_in_zone(*zone) {
                        if self.passes_filters(state, entity, player_count) {
                            targets.push(entity);
                        }
                    }
                }
            }
            TargetType::Any => {
                // Players
                for player in PlayerId::all(player_count) {
                    let entity = EntityId::player(player);
                    if self.passes_filters(state, entity, player_count) {
                        targets.push(entity);
                    }
                }
                // All cards (would need zone enumeration from game config)
                // For now, Any requires explicit zone list in practice
            }
        }

        targets
    }

    /// Check if an entity passes all filters.
    fn passes_filters(&self, state: &GameState, entity: EntityId, player_count: usize) -> bool {
        for filter in &self.spec.filters {
            if !self.passes_filter(state, entity, filter, player_count) {
                return false;
            }
        }
        true
    }

    /// Check if an entity passes a single filter.
    fn passes_filter(
        &self,
        state: &GameState,
        entity: EntityId,
        filter: &TargetFilter,
        player_count: usize,
    ) -> bool {
        match filter {
            TargetFilter::Opponent => {
                entity.as_player_index(player_count).is_some_and(|idx| {
                    PlayerId::new(idx) != self.acting_player
                })
            }
            TargetFilter::Self_ => {
                entity.as_player_index(player_count).is_some_and(|idx| {
                    PlayerId::new(idx) == self.acting_player
                })
            }
            TargetFilter::NotSelf => {
                entity.as_player_index(player_count).is_none_or(|idx| {
                    PlayerId::new(idx) != self.acting_player
                })
            }
            TargetFilter::SpecificPlayer(player) => {
                entity.as_player_index(player_count).is_some_and(|idx| {
                    PlayerId::new(idx) == *player
                })
            }
            TargetFilter::InZone(zone) => {
                state.zones.is_in_zone(entity, *zone)
            }
            TargetFilter::HasCardType(_card_type) => {
                // Requires CardRegistry access which TargetSelector doesn't have.
                // Return false (conservative) - games should use Custom filters
                // with registry access for card type filtering.
                false
            }
            TargetFilter::OwnedBy(player) => {
                state.get_card(entity).is_some_and(|card| {
                    card.owner == Some(*player)
                })
            }
            TargetFilter::ControlledBy(player) => {
                state.get_card(entity).is_some_and(|card| {
                    card.controller == Some(*player)
                })
            }
            TargetFilter::StateInRange { key, min, max } => {
                if entity.is_player(player_count) {
                    let idx = entity.as_player_index(player_count).unwrap();
                    let value = state.public.get_player_state(PlayerId::new(idx), key, 0);
                    min.is_none_or(|m| value >= m) && max.is_none_or(|m| value <= m)
                } else if let Some(card) = state.get_card(entity) {
                    let value = card.get_state(key, 0);
                    min.is_none_or(|m| value >= m) && max.is_none_or(|m| value <= m)
                } else {
                    false
                }
            }
            TargetFilter::NotSource => {
                self.source_entity != Some(entity)
            }
            TargetFilter::Custom(_) => {
                // Custom filters must be handled by game-specific code
                true
            }
        }
    }

    /// Check if the minimum target count can be satisfied.
    pub fn has_enough_targets(&self, state: &GameState) -> bool {
        let targets = self.valid_targets(state);
        let min_required = match &self.spec.count {
            TargetCount::Exactly(n) => *n,
            TargetCount::AtLeast(n) => *n,
            TargetCount::Range { min, .. } => *min,
            TargetCount::UpTo(_) => 0,
            TargetCount::All => 0,
        };

        if self.spec.optional {
            true
        } else {
            targets.len() >= min_required
        }
    }

    /// Validate a selection of targets.
    pub fn validate_selection(&self, state: &GameState, selected: &[EntityId]) -> bool {
        // Check count
        let count_valid = match &self.spec.count {
            TargetCount::Exactly(n) => selected.len() == *n,
            TargetCount::UpTo(n) => selected.len() <= *n,
            TargetCount::AtLeast(n) => selected.len() >= *n,
            TargetCount::Range { min, max } => selected.len() >= *min && selected.len() <= *max,
            TargetCount::All => true,
        };

        if !(count_valid || self.spec.optional && selected.is_empty()) {
            return false;
        }

        // Check all selected are valid
        let valid_targets = self.valid_targets(state);
        selected.iter().all(|s| valid_targets.contains(s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_spec_single_player() {
        let spec = TargetSpec::single_player();
        assert_eq!(spec.count, TargetCount::Exactly(1));
        assert!(!spec.optional);
    }

    #[test]
    fn test_target_spec_opponent() {
        let spec = TargetSpec::single_opponent();
        assert!(spec.filters.contains(&TargetFilter::Opponent));
    }

    #[test]
    fn test_target_spec_builder() {
        let zone = ZoneId::new(0);
        let spec = TargetSpec::single_card([zone])
            .with_filter(TargetFilter::OwnedBy(PlayerId::new(0)))
            .optional();

        assert!(spec.optional);
        assert_eq!(spec.filters.len(), 1);
    }

    #[test]
    fn test_target_selector_player() {
        let state = GameState::new(4, 42);

        let spec = TargetSpec::single_opponent();
        let selector = TargetSelector::new(spec, PlayerId::new(0));

        let targets = selector.valid_targets(&state);

        // Should have 3 opponents (players 1, 2, 3)
        assert_eq!(targets.len(), 3);
        assert!(!targets.contains(&EntityId::player_id(0)));
        assert!(targets.contains(&EntityId::player_id(1)));
        assert!(targets.contains(&EntityId::player_id(2)));
        assert!(targets.contains(&EntityId::player_id(3)));
    }

    #[test]
    fn test_target_selector_self() {
        let state = GameState::new(2, 42);

        let spec = TargetSpec::single_player()
            .with_filter(TargetFilter::Self_);
        let selector = TargetSelector::new(spec, PlayerId::new(1));

        let targets = selector.valid_targets(&state);

        assert_eq!(targets.len(), 1);
        assert!(targets.contains(&EntityId::player_id(1)));
    }

    #[test]
    fn test_validate_selection() {
        let state = GameState::new(2, 42);

        let spec = TargetSpec::single_opponent();
        let selector = TargetSelector::new(spec, PlayerId::new(0));

        // Valid: selecting player 1 (opponent)
        assert!(selector.validate_selection(&state, &[EntityId::player_id(1)]));

        // Invalid: selecting self
        assert!(!selector.validate_selection(&state, &[EntityId::player_id(0)]));

        // Invalid: selecting two targets when only one allowed
        assert!(!selector.validate_selection(&state, &[EntityId::player_id(1), EntityId::player_id(0)]));
    }

    #[test]
    fn test_optional_targeting() {
        let state = GameState::new(1, 42);

        let spec = TargetSpec::single_opponent().optional();
        let selector = TargetSelector::new(spec, PlayerId::new(0));

        // No opponents in 1-player game
        assert!(selector.valid_targets(&state).is_empty());

        // But optional, so empty selection is valid
        assert!(selector.validate_selection(&state, &[]));
        assert!(selector.has_enough_targets(&state));
    }
}

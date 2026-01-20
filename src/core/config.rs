//! Game configuration types.
//!
//! Games configure the engine at startup by providing:
//! - `ZoneConfig`: Defines zones (hand, library, battlefield, etc.)
//! - `TemplateConfig`: Defines action types (play card, attack, etc.)
//! - `GameConfig`: Combines all configuration
//!
//! The engine never hardcodes zones or action types - games define them.

use serde::{Deserialize, Serialize};

use super::PlayerId;

/// Zone identifier. Games define what zones exist.
///
/// The engine doesn't interpret zone IDs - they're opaque identifiers.
/// Games assign meaning via `ZoneConfig`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ZoneId(pub u16);

impl ZoneId {
    /// Create a new zone ID.
    #[must_use]
    pub const fn new(id: u16) -> Self {
        Self(id)
    }

    /// Get the raw ID value.
    #[must_use]
    pub const fn raw(self) -> u16 {
        self.0
    }
}

impl std::fmt::Display for ZoneId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Zone({})", self.0)
    }
}

/// Zone visibility rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZoneVisibility {
    /// All cards visible to all players (battlefield).
    Public,
    /// Cards visible only to the zone owner (hand).
    OwnerOnly,
    /// Cards not visible to anyone (face-down library).
    Hidden,
    /// Custom visibility rules handled by game-specific code.
    ///
    /// Games using this should implement their own visibility logic
    /// based on game state (e.g., cards revealed by effects).
    Custom,
}

/// Configuration for a single zone.
///
/// Games define their zones at startup. The engine uses these configs
/// to determine visibility, ordering, and other zone properties.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZoneConfig {
    /// Unique identifier for this zone.
    pub id: ZoneId,

    /// Human-readable name (for debugging/display).
    pub name: String,

    /// Zone owner. `None` for shared zones (battlefield, market).
    pub owner: Option<PlayerId>,

    /// Visibility rules for cards in this zone.
    pub visibility: ZoneVisibility,

    /// Is card order significant? (true for library, stack, false for battlefield).
    pub ordered: bool,

    /// Maximum cards allowed. `None` for unlimited.
    pub max_cards: Option<usize>,
}

impl ZoneConfig {
    /// Create a new zone configuration.
    pub fn new(id: ZoneId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            owner: None,
            visibility: ZoneVisibility::Public,
            ordered: false,
            max_cards: None,
        }
    }

    /// Set the zone owner.
    #[must_use]
    pub fn with_owner(mut self, owner: PlayerId) -> Self {
        self.owner = Some(owner);
        self
    }

    /// Set visibility to owner-only (like a hand).
    #[must_use]
    pub fn owner_only(mut self) -> Self {
        self.visibility = ZoneVisibility::OwnerOnly;
        self
    }

    /// Set visibility to hidden (like a face-down library).
    #[must_use]
    pub fn hidden(mut self) -> Self {
        self.visibility = ZoneVisibility::Hidden;
        self
    }

    /// Mark zone as ordered (like library or stack).
    #[must_use]
    pub fn ordered(mut self) -> Self {
        self.ordered = true;
        self
    }

    /// Set maximum card limit.
    #[must_use]
    pub fn with_max_cards(mut self, max: usize) -> Self {
        self.max_cards = Some(max);
        self
    }
}

/// Action template identifier. Games define what action types exist.
///
/// The engine doesn't interpret template IDs - they're opaque identifiers.
/// Games assign meaning via `TemplateConfig`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TemplateId(pub u16);

impl TemplateId {
    /// Create a new template ID.
    #[must_use]
    pub const fn new(id: u16) -> Self {
        Self(id)
    }

    /// Get the raw ID value.
    #[must_use]
    pub const fn raw(self) -> u16 {
        self.0
    }
}

impl std::fmt::Display for TemplateId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Template({})", self.0)
    }
}

/// Configuration for an action template.
///
/// Games define their action types at startup. Each template specifies
/// how many entity pointers the action requires.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemplateConfig {
    /// Unique identifier for this template.
    pub id: TemplateId,

    /// Human-readable name (for debugging/display).
    pub name: String,

    /// Number of entity pointers required.
    ///
    /// Examples:
    /// - "Pass": 0 pointers
    /// - "Play card": 1 pointer (the card)
    /// - "Attack": 1 pointer (the attacker)
    /// - "Block": 2 pointers (blocker, attacker being blocked)
    /// - "Cast spell on target": 2 pointers (spell, target)
    pub pointer_count: usize,

    /// Can this action have additional variable pointers?
    ///
    /// True for actions like "multi-target spell" where the number
    /// of targets varies.
    pub variable_pointers: bool,
}

impl TemplateConfig {
    /// Create a new template configuration.
    pub fn new(id: TemplateId, name: impl Into<String>, pointer_count: usize) -> Self {
        Self {
            id,
            name: name.into(),
            pointer_count,
            variable_pointers: false,
        }
    }

    /// Create a template with no pointers (like "Pass").
    pub fn no_args(id: TemplateId, name: impl Into<String>) -> Self {
        Self::new(id, name, 0)
    }

    /// Allow variable additional pointers.
    #[must_use]
    pub fn with_variable_pointers(mut self) -> Self {
        self.variable_pointers = true;
        self
    }
}

/// Opaque phase identifier. Games define their own phases.
///
/// The engine doesn't interpret phase IDs - they're just compared
/// for equality. Games define phase transitions in their rules.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PhaseId(pub u32);

impl PhaseId {
    /// Create a new phase ID.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }
}

/// Complete game configuration.
///
/// Games provide this at startup to configure the engine.
#[derive(Clone, Debug)]
pub struct GameConfig {
    /// Number of players (1-255).
    pub player_count: usize,

    /// Zone configurations.
    pub zones: Vec<ZoneConfig>,

    /// Action template configurations.
    pub templates: Vec<TemplateConfig>,

    /// Initial game phase.
    pub initial_phase: PhaseId,
}

impl GameConfig {
    /// Create a new game configuration.
    pub fn new(player_count: usize) -> Self {
        assert!(player_count > 0, "Must have at least 1 player");
        assert!(player_count <= 255, "At most 255 players supported");

        Self {
            player_count,
            zones: Vec::new(),
            templates: Vec::new(),
            initial_phase: PhaseId::default(),
        }
    }

    /// Add a zone configuration.
    #[must_use]
    pub fn with_zone(mut self, zone: ZoneConfig) -> Self {
        self.zones.push(zone);
        self
    }

    /// Add a template configuration.
    #[must_use]
    pub fn with_template(mut self, template: TemplateConfig) -> Self {
        self.templates.push(template);
        self
    }

    /// Set the initial phase.
    #[must_use]
    pub fn with_initial_phase(mut self, phase: PhaseId) -> Self {
        self.initial_phase = phase;
        self
    }

    /// Get a zone config by ID.
    #[must_use]
    pub fn get_zone(&self, id: ZoneId) -> Option<&ZoneConfig> {
        self.zones.iter().find(|z| z.id == id)
    }

    /// Get a template config by ID.
    #[must_use]
    pub fn get_template(&self, id: TemplateId) -> Option<&TemplateConfig> {
        self.templates.iter().find(|t| t.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zone_id() {
        let id = ZoneId::new(5);
        assert_eq!(id.raw(), 5);
        assert_eq!(format!("{}", id), "Zone(5)");
    }

    #[test]
    fn test_zone_config_builder() {
        let zone = ZoneConfig::new(ZoneId::new(0), "Hand")
            .with_owner(PlayerId::new(0))
            .owner_only()
            .with_max_cards(7);

        assert_eq!(zone.name, "Hand");
        assert_eq!(zone.owner, Some(PlayerId::new(0)));
        assert_eq!(zone.visibility, ZoneVisibility::OwnerOnly);
        assert_eq!(zone.max_cards, Some(7));
        assert!(!zone.ordered);
    }

    #[test]
    fn test_zone_config_ordered() {
        let zone = ZoneConfig::new(ZoneId::new(1), "Library")
            .with_owner(PlayerId::new(0))
            .hidden()
            .ordered();

        assert!(zone.ordered);
        assert_eq!(zone.visibility, ZoneVisibility::Hidden);
    }

    #[test]
    fn test_template_id() {
        let id = TemplateId::new(3);
        assert_eq!(id.raw(), 3);
        assert_eq!(format!("{}", id), "Template(3)");
    }

    #[test]
    fn test_template_config() {
        let pass = TemplateConfig::no_args(TemplateId::new(0), "Pass");
        assert_eq!(pass.pointer_count, 0);
        assert!(!pass.variable_pointers);

        let cast = TemplateConfig::new(TemplateId::new(1), "Cast", 2)
            .with_variable_pointers();
        assert_eq!(cast.pointer_count, 2);
        assert!(cast.variable_pointers);
    }

    #[test]
    fn test_game_config() {
        let config = GameConfig::new(2)
            .with_zone(ZoneConfig::new(ZoneId::new(0), "Battlefield"))
            .with_zone(ZoneConfig::new(ZoneId::new(1), "Hand").owner_only())
            .with_template(TemplateConfig::no_args(TemplateId::new(0), "Pass"))
            .with_initial_phase(PhaseId::new(1));

        assert_eq!(config.player_count, 2);
        assert_eq!(config.zones.len(), 2);
        assert_eq!(config.templates.len(), 1);
        assert_eq!(config.initial_phase, PhaseId::new(1));

        assert!(config.get_zone(ZoneId::new(0)).is_some());
        assert!(config.get_zone(ZoneId::new(99)).is_none());
        assert!(config.get_template(TemplateId::new(0)).is_some());
    }

    #[test]
    #[should_panic(expected = "Must have at least 1 player")]
    fn test_game_config_zero_players() {
        GameConfig::new(0);
    }

    #[test]
    fn test_phase_id() {
        let phase = PhaseId::new(5);
        assert_eq!(phase.0, 5);

        let default = PhaseId::default();
        assert_eq!(default.0, 0);
    }
}

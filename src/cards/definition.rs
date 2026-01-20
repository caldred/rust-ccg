//! Card definitions - static card data.
//!
//! `CardDefinition` holds the immutable properties of a card type.
//! For example, "Lightning Bolt" has a cost of R and deals 3 damage -
//! these are part of the definition.
//!
//! Instance-specific data (damage taken, counters, zone) is stored
//! separately in `CardInstance`.

use serde::{Deserialize, Serialize};

use super::attributes::{AttributeKey, AttributeValue, Attributes};

/// Unique identifier for a card definition.
///
/// This identifies the "type" of card (e.g., "Lightning Bolt"),
/// not a specific instance in a game.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CardId(pub u32);

impl CardId {
    /// Create a new card ID.
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

impl std::fmt::Display for CardId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Card({})", self.0)
    }
}

/// Card type identifier - games define their own types.
///
/// The engine doesn't interpret these. Games define what types exist
/// (Creature, Spell, Land, etc.) and assign meaning.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CardTypeId(pub u32);

impl CardTypeId {
    /// Create a new card type ID.
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

/// Static card definition.
///
/// Contains the unchanging data about a card type.
/// All game-specific data goes in `attributes`.
///
/// ## Example
///
/// ```
/// use rust_ccg::cards::{CardDefinition, CardId, CardTypeId};
///
/// let bolt = CardDefinition::new(CardId::new(1), "Lightning Bolt", CardTypeId::new(0))
///     .with_attr("cost", 1i32)
///     .with_attr("damage", 3i32);
///
/// assert_eq!(bolt.get_int("damage", 0), 3);
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CardDefinition {
    /// Unique identifier for this card definition.
    pub id: CardId,

    /// Card name (for display/debugging).
    pub name: String,

    /// Card type (game-specific, opaque to engine).
    pub card_type: CardTypeId,

    /// Game-specific attributes.
    pub attributes: Attributes,
}

impl CardDefinition {
    /// Create a new card definition.
    #[must_use]
    pub fn new(id: CardId, name: impl Into<String>, card_type: CardTypeId) -> Self {
        Self {
            id,
            name: name.into(),
            card_type,
            attributes: Attributes::default(),
        }
    }

    /// Add an attribute (builder pattern).
    #[must_use]
    pub fn with_attr(
        mut self,
        key: impl Into<AttributeKey>,
        value: impl Into<AttributeValue>,
    ) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Get an attribute value.
    #[must_use]
    pub fn get_attr(&self, key: &str) -> Option<&AttributeValue> {
        self.attributes.get(&AttributeKey::new(key))
    }

    /// Get an integer attribute with a default value.
    #[must_use]
    pub fn get_int(&self, key: &str, default: i64) -> i64 {
        self.get_attr(key)
            .and_then(|v| v.as_int())
            .unwrap_or(default)
    }

    /// Get a boolean attribute with a default value.
    #[must_use]
    pub fn get_bool(&self, key: &str, default: bool) -> bool {
        self.get_attr(key)
            .and_then(|v| v.as_bool())
            .unwrap_or(default)
    }

    /// Get a text attribute.
    #[must_use]
    pub fn get_text(&self, key: &str) -> Option<&str> {
        self.get_attr(key).and_then(|v| v.as_text())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_card_id() {
        let id = CardId::new(42);
        assert_eq!(id.raw(), 42);
        assert_eq!(format!("{}", id), "Card(42)");
    }

    #[test]
    fn test_card_type_id() {
        let id = CardTypeId::new(1);
        assert_eq!(id.raw(), 1);
    }

    #[test]
    fn test_card_definition_builder() {
        let card = CardDefinition::new(CardId::new(1), "Test Card", CardTypeId::new(0))
            .with_attr("cost", 3i32)
            .with_attr("power", 2i32)
            .with_attr("toughness", 2i32)
            .with_attr("flying", true);

        assert_eq!(card.name, "Test Card");
        assert_eq!(card.id, CardId::new(1));
        assert_eq!(card.get_int("cost", 0), 3);
        assert_eq!(card.get_int("power", 0), 2);
        assert_eq!(card.get_bool("flying", false), true);
        assert_eq!(card.get_bool("trample", false), false); // default
    }

    #[test]
    fn test_card_definition_text_attr() {
        let card = CardDefinition::new(CardId::new(1), "Test", CardTypeId::new(0))
            .with_attr("subtype", "Goblin");

        assert_eq!(card.get_text("subtype"), Some("Goblin"));
        assert_eq!(card.get_text("missing"), None);
    }

    #[test]
    fn test_card_definition_serialization() {
        let card = CardDefinition::new(CardId::new(1), "Test", CardTypeId::new(0))
            .with_attr("cost", 2i32);

        let json = serde_json::to_string(&card).unwrap();
        let deserialized: CardDefinition = serde_json::from_str(&json).unwrap();

        assert_eq!(card.id, deserialized.id);
        assert_eq!(card.name, deserialized.name);
    }
}

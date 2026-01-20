//! Card registry for definition lookup.
//!
//! The `CardRegistry` stores all card definitions for a game.
//! It provides fast lookup by `CardId` and supports iteration.

use rustc_hash::FxHashMap;

use super::definition::{CardDefinition, CardId, CardTypeId};

/// Registry of card definitions.
///
/// Stores all card definitions for a game and provides lookup.
///
/// ## Example
///
/// ```
/// use rust_ccg::cards::{CardRegistry, CardDefinition, CardId, CardTypeId};
///
/// let mut registry = CardRegistry::new();
///
/// let bolt = CardDefinition::new(CardId::new(1), "Lightning Bolt", CardTypeId::new(0))
///     .with_attr("damage", 3i32);
///
/// registry.register(bolt);
///
/// let found = registry.get(CardId::new(1)).unwrap();
/// assert_eq!(found.name, "Lightning Bolt");
/// ```
#[derive(Clone, Debug, Default)]
pub struct CardRegistry {
    cards: FxHashMap<CardId, CardDefinition>,
    next_id: u32,
}

impl CardRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a card definition.
    ///
    /// Panics if a card with the same ID already exists.
    pub fn register(&mut self, card: CardDefinition) {
        if self.cards.contains_key(&card.id) {
            panic!("Card with ID {:?} already registered", card.id);
        }
        self.cards.insert(card.id, card);
    }

    /// Register a card and return a mutable reference.
    ///
    /// Useful for building cards incrementally.
    pub fn register_mut(&mut self, card: CardDefinition) -> &mut CardDefinition {
        let id = card.id;
        self.register(card);
        self.cards.get_mut(&id).unwrap()
    }

    /// Register a card with an auto-assigned ID.
    ///
    /// Returns the assigned ID.
    pub fn register_auto(&mut self, name: impl Into<String>, card_type: CardTypeId) -> CardId {
        let id = CardId::new(self.next_id);
        self.next_id += 1;

        let card = CardDefinition::new(id, name, card_type);
        self.register(card);
        id
    }

    /// Get a card definition by ID.
    #[must_use]
    pub fn get(&self, id: CardId) -> Option<&CardDefinition> {
        self.cards.get(&id)
    }

    /// Get a card definition by ID, panicking if not found.
    ///
    /// Use when you're certain the card exists.
    #[must_use]
    pub fn get_unchecked(&self, id: CardId) -> &CardDefinition {
        self.cards.get(&id).expect("Card not found in registry")
    }

    /// Check if a card ID is registered.
    #[must_use]
    pub fn contains(&self, id: CardId) -> bool {
        self.cards.contains_key(&id)
    }

    /// Get the number of registered cards.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cards.len()
    }

    /// Check if the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cards.is_empty()
    }

    /// Iterate over all card definitions.
    pub fn iter(&self) -> impl Iterator<Item = &CardDefinition> {
        self.cards.values()
    }

    /// Find cards by type.
    pub fn find_by_type(&self, card_type: CardTypeId) -> impl Iterator<Item = &CardDefinition> {
        self.cards.values().filter(move |c| c.card_type == card_type)
    }

    /// Find cards matching a predicate.
    pub fn find<F>(&self, predicate: F) -> impl Iterator<Item = &CardDefinition>
    where
        F: Fn(&CardDefinition) -> bool,
    {
        self.cards.values().filter(move |c| predicate(c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_get() {
        let mut registry = CardRegistry::new();

        let card = CardDefinition::new(CardId::new(1), "Test Card", CardTypeId::new(0));
        registry.register(card);

        let found = registry.get(CardId::new(1));
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "Test Card");

        assert!(registry.get(CardId::new(99)).is_none());
    }

    #[test]
    fn test_register_auto() {
        let mut registry = CardRegistry::new();

        let id1 = registry.register_auto("Card A", CardTypeId::new(0));
        let id2 = registry.register_auto("Card B", CardTypeId::new(0));

        assert_eq!(id1, CardId::new(0));
        assert_eq!(id2, CardId::new(1));
        assert_eq!(registry.len(), 2);
    }

    #[test]
    #[should_panic(expected = "already registered")]
    fn test_duplicate_id_panics() {
        let mut registry = CardRegistry::new();

        let card1 = CardDefinition::new(CardId::new(1), "Card A", CardTypeId::new(0));
        let card2 = CardDefinition::new(CardId::new(1), "Card B", CardTypeId::new(0));

        registry.register(card1);
        registry.register(card2); // Should panic
    }

    #[test]
    fn test_find_by_type() {
        let mut registry = CardRegistry::new();

        let creature_type = CardTypeId::new(0);
        let spell_type = CardTypeId::new(1);

        registry.register(CardDefinition::new(CardId::new(1), "Goblin", creature_type));
        registry.register(CardDefinition::new(CardId::new(2), "Bolt", spell_type));
        registry.register(CardDefinition::new(CardId::new(3), "Orc", creature_type));

        let creatures: Vec<_> = registry.find_by_type(creature_type).collect();
        assert_eq!(creatures.len(), 2);

        let spells: Vec<_> = registry.find_by_type(spell_type).collect();
        assert_eq!(spells.len(), 1);
    }

    #[test]
    fn test_find_with_predicate() {
        let mut registry = CardRegistry::new();

        registry.register(
            CardDefinition::new(CardId::new(1), "Cheap", CardTypeId::new(0))
                .with_attr("cost", 1i32),
        );
        registry.register(
            CardDefinition::new(CardId::new(2), "Expensive", CardTypeId::new(0))
                .with_attr("cost", 5i32),
        );

        let cheap: Vec<_> = registry.find(|c| c.get_int("cost", 0) <= 2).collect();
        assert_eq!(cheap.len(), 1);
        assert_eq!(cheap[0].name, "Cheap");
    }

    #[test]
    fn test_iteration() {
        let mut registry = CardRegistry::new();

        registry.register(CardDefinition::new(CardId::new(1), "A", CardTypeId::new(0)));
        registry.register(CardDefinition::new(CardId::new(2), "B", CardTypeId::new(0)));

        let names: Vec<_> = registry.iter().map(|c| &c.name).collect();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&&"A".to_string()));
        assert!(names.contains(&&"B".to_string()));
    }

    #[test]
    fn test_contains() {
        let mut registry = CardRegistry::new();
        registry.register(CardDefinition::new(CardId::new(1), "Test", CardTypeId::new(0)));

        assert!(registry.contains(CardId::new(1)));
        assert!(!registry.contains(CardId::new(99)));
    }
}

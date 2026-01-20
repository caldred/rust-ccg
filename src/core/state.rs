//! Game state: public and private information.
//!
//! ## PublicState
//!
//! Observable information for all players:
//! - Phase, turn, active player, priority
//! - Player state (life totals, resources)
//! - Hand sizes, known cards
//! - Action history
//!
//! ## GameState
//!
//! Complete game state including:
//! - Public state
//! - Zone manager (card locations)
//! - Private hands and decks
//! - RNG

use im::{HashSet as ImHashSet, Vector};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use super::action::ActionRecord;
use super::config::PhaseId;
use super::player::{PlayerId, PlayerMap};
use super::rng::GameRng;
use crate::cards::{CardId, CardInstance};
use crate::zones::ZoneManager;

/// Public game state - observable by all players.
///
/// Uses `im` persistent data structures for O(1) cloning in MCTS.
///
/// ## State Values (i64 only)
///
/// `player_state` and `turn_state` use `FxHashMap<String, i64>` for performance:
/// - Fast hashing for MCTS
/// - Efficient cloning
///
/// To store non-integer values:
/// - Booleans: use 0/1
/// - Entity references: use EntityId.0 as i64
/// - Enums: use discriminant values
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublicState {
    // === Configuration ===
    player_count: usize,

    // === Game Progression ===
    /// Current phase (game-specific, opaque to engine).
    pub phase: PhaseId,

    /// Turn number (starts at 1).
    pub turn_number: u32,

    /// Action sequence within turn.
    pub action_sequence: u32,

    /// Active player (whose turn it is).
    pub active_player: PlayerId,

    /// Players with priority (can act).
    /// Most games: single player. Simultaneous: multiple.
    pub priority_players: Vec<PlayerId>,

    // === Player State ===
    /// Per-player state (life, mana, etc.) - games define keys.
    pub player_state: PlayerMap<FxHashMap<String, i64>>,

    /// Per-turn state (cleared on turn advance by default).
    pub turn_state: FxHashMap<String, i64>,

    // === Information Tracking ===
    /// Hand sizes (public knowledge).
    pub hand_sizes: PlayerMap<u32>,

    /// Known cards in hands (revealed via effects).
    pub known_hand_cards: PlayerMap<ImHashSet<CardId>>,

    /// Action history for MCTS opponent consistency.
    pub action_history: Vector<ActionRecord>,
}

impl PublicState {
    /// Create a new public state.
    ///
    /// ## Defaults
    ///
    /// - `active_player`: Player 0
    /// - `priority_players`: [Player 0]
    /// - `phase`: PhaseId(0)
    /// - `turn_number`: 1
    ///
    /// Call configuration methods to customize, especially for:
    /// - Random starting player
    /// - Simultaneous games (all players have priority)
    #[must_use]
    pub fn new(player_count: usize) -> Self {
        assert!(player_count > 0, "Must have at least 1 player");
        assert!(player_count <= 255, "At most 255 players supported");

        Self {
            player_count,
            phase: PhaseId::default(),
            turn_number: 1,
            action_sequence: 0,
            active_player: PlayerId::new(0),
            priority_players: vec![PlayerId::new(0)],
            player_state: PlayerMap::with_default(player_count),
            turn_state: FxHashMap::default(),
            hand_sizes: PlayerMap::with_value(player_count, 0),
            known_hand_cards: PlayerMap::new(player_count, |_| ImHashSet::new()),
            action_history: Vector::new(),
        }
    }

    /// Get player count.
    #[must_use]
    pub fn player_count(&self) -> usize {
        self.player_count
    }

    /// Iterate over all player IDs.
    pub fn player_ids(&self) -> impl Iterator<Item = PlayerId> {
        PlayerId::all(self.player_count)
    }

    // === Player State ===

    /// Get a player state value with default.
    #[must_use]
    pub fn get_player_state(&self, player: PlayerId, key: &str, default: i64) -> i64 {
        self.player_state[player].get(key).copied().unwrap_or(default)
    }

    /// Set a player state value.
    pub fn set_player_state(&mut self, player: PlayerId, key: impl Into<String>, value: i64) {
        self.player_state[player].insert(key.into(), value);
    }

    /// Modify a player state value by delta.
    pub fn modify_player_state(&mut self, player: PlayerId, key: &str, delta: i64) {
        let current = self.get_player_state(player, key, 0);
        self.player_state[player].insert(key.to_string(), current + delta);
    }

    // === Turn State ===

    /// Get a turn state value with default.
    #[must_use]
    pub fn get_turn_state(&self, key: &str, default: i64) -> i64 {
        self.turn_state.get(key).copied().unwrap_or(default)
    }

    /// Set a turn state value.
    pub fn set_turn_state(&mut self, key: impl Into<String>, value: i64) {
        self.turn_state.insert(key.into(), value);
    }

    // === Priority ===

    /// Set the active player.
    pub fn set_active_player(&mut self, player: PlayerId) {
        self.active_player = player;
    }

    /// Set priority to a single player.
    pub fn set_priority(&mut self, player: PlayerId) {
        self.priority_players = vec![player];
    }

    /// Set priority to multiple players (simultaneous games).
    pub fn set_priority_multiple(&mut self, players: Vec<PlayerId>) {
        self.priority_players = players;
    }

    /// Check if a player has priority.
    #[must_use]
    pub fn has_priority(&self, player: PlayerId) -> bool {
        self.priority_players.contains(&player)
    }

    // === Turn Advancement ===

    /// Advance to next turn, clearing turn_state.
    pub fn advance_turn(&mut self) {
        self.turn_number += 1;
        self.turn_state.clear();
        self.action_sequence = 0;
    }

    /// Advance to next turn, preserving turn_state.
    pub fn advance_turn_preserve_state(&mut self) {
        self.turn_number += 1;
        self.action_sequence = 0;
    }

    // === Action History ===

    /// Record an action in history.
    pub fn record_action(&mut self, record: ActionRecord) {
        self.action_history.push_back(record);
    }

    /// Get the next action sequence number and increment.
    pub fn next_sequence(&mut self) -> u32 {
        let seq = self.action_sequence;
        self.action_sequence += 1;
        seq
    }
}

/// Full game state including private information.
pub struct GameState {
    /// Public state (observable by all).
    pub public: PublicState,

    /// Zone manager for card locations.
    pub zones: ZoneManager,

    /// Private hands per player.
    hands: PlayerMap<Vec<CardId>>,

    /// Private decks per player (top = end of vec).
    decks: PlayerMap<Vec<CardId>>,

    /// Card instances by entity ID.
    cards: FxHashMap<crate::core::EntityId, CardInstance>,

    /// Deterministic RNG.
    pub rng: GameRng,

    /// Next entity ID to allocate.
    next_entity_id: u32,
}

impl GameState {
    /// Create a new game state.
    #[must_use]
    pub fn new(player_count: usize, seed: u64) -> Self {
        Self {
            public: PublicState::new(player_count),
            zones: ZoneManager::new(),
            hands: PlayerMap::with_default(player_count),
            decks: PlayerMap::with_default(player_count),
            cards: FxHashMap::default(),
            rng: GameRng::new(seed),
            next_entity_id: crate::core::EntityId::first_non_player(player_count),
        }
    }

    /// Get player count.
    #[must_use]
    pub fn player_count(&self) -> usize {
        self.public.player_count()
    }

    // === Entity Management ===

    /// Allocate a new entity ID.
    pub fn alloc_entity(&mut self) -> crate::core::EntityId {
        let id = crate::core::EntityId(self.next_entity_id);
        self.next_entity_id += 1;
        id
    }

    /// Add a card instance.
    pub fn add_card(&mut self, card: CardInstance) {
        let entity_id = card.entity_id;
        let zone = card.zone;
        self.cards.insert(entity_id, card);
        self.zones.add_to_zone(entity_id, zone, None);
    }

    /// Get a card instance.
    #[must_use]
    pub fn get_card(&self, entity_id: crate::core::EntityId) -> Option<&CardInstance> {
        self.cards.get(&entity_id)
    }

    /// Get a mutable card instance.
    pub fn get_card_mut(&mut self, entity_id: crate::core::EntityId) -> Option<&mut CardInstance> {
        self.cards.get_mut(&entity_id)
    }

    // === Hands ===

    /// Get a player's hand.
    #[must_use]
    pub fn hand(&self, player: PlayerId) -> &[CardId] {
        &self.hands[player]
    }

    /// Add a card to a player's hand.
    pub fn add_to_hand(&mut self, player: PlayerId, card_id: CardId) {
        self.hands[player].push(card_id);
        self.public.hand_sizes[player] += 1;
    }

    /// Remove a card from a player's hand.
    ///
    /// Returns true if the card was found and removed.
    pub fn remove_from_hand(&mut self, player: PlayerId, card_id: CardId) -> bool {
        if let Some(pos) = self.hands[player].iter().position(|&c| c == card_id) {
            self.hands[player].remove(pos);
            self.public.hand_sizes[player] -= 1;
            true
        } else {
            false
        }
    }

    // === Decks ===

    /// Set a player's deck.
    pub fn set_deck(&mut self, player: PlayerId, deck: Vec<CardId>) {
        self.decks[player] = deck;
    }

    /// Get a player's deck.
    #[must_use]
    pub fn deck(&self, player: PlayerId) -> &[CardId] {
        &self.decks[player]
    }

    /// Get deck size.
    #[must_use]
    pub fn deck_size(&self, player: PlayerId) -> usize {
        self.decks[player].len()
    }

    /// Draw a card from a player's deck to hand.
    ///
    /// Returns the drawn card ID, or None if deck is empty.
    pub fn draw_card(&mut self, player: PlayerId) -> Option<CardId> {
        let card_id = self.decks[player].pop()?;
        self.add_to_hand(player, card_id);
        Some(card_id)
    }

    /// Shuffle a player's deck.
    pub fn shuffle_deck(&mut self, player: PlayerId) {
        self.rng.shuffle(&mut self.decks[player]);
    }

    // === Cloning ===

    /// Clone the game state (for MCTS).
    ///
    /// Uses persistent data structures where possible for efficiency.
    /// Takes `&mut self` because forking the RNG advances the fork counter.
    #[must_use]
    pub fn clone_state(&mut self) -> Self {
        Self {
            public: self.public.clone(),
            zones: self.zones.clone(),
            hands: self.hands.clone(),
            decks: self.decks.clone(),
            cards: self.cards.clone(),
            rng: self.rng.fork(),
            next_entity_id: self.next_entity_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_public_state_new() {
        let state = PublicState::new(4);

        assert_eq!(state.player_count(), 4);
        assert_eq!(state.turn_number, 1);
        assert_eq!(state.active_player, PlayerId::new(0));
        assert!(state.has_priority(PlayerId::new(0)));
    }

    #[test]
    fn test_player_state() {
        let mut state = PublicState::new(2);

        assert_eq!(state.get_player_state(PlayerId::new(0), "life", 20), 20);

        state.set_player_state(PlayerId::new(0), "life", 15);
        assert_eq!(state.get_player_state(PlayerId::new(0), "life", 20), 15);

        state.modify_player_state(PlayerId::new(0), "life", -3);
        assert_eq!(state.get_player_state(PlayerId::new(0), "life", 20), 12);
    }

    #[test]
    fn test_turn_advance() {
        let mut state = PublicState::new(2);
        state.set_turn_state("land_played", 1);

        state.advance_turn();

        assert_eq!(state.turn_number, 2);
        assert_eq!(state.get_turn_state("land_played", 0), 0); // Cleared
    }

    #[test]
    fn test_turn_advance_preserve() {
        let mut state = PublicState::new(2);
        state.set_turn_state("resource", 5);

        state.advance_turn_preserve_state();

        assert_eq!(state.turn_number, 2);
        assert_eq!(state.get_turn_state("resource", 0), 5); // Preserved
    }

    #[test]
    fn test_priority_multiple() {
        let mut state = PublicState::new(4);

        state.set_priority_multiple(vec![
            PlayerId::new(0),
            PlayerId::new(1),
            PlayerId::new(2),
            PlayerId::new(3),
        ]);

        assert!(state.has_priority(PlayerId::new(0)));
        assert!(state.has_priority(PlayerId::new(1)));
        assert!(state.has_priority(PlayerId::new(2)));
        assert!(state.has_priority(PlayerId::new(3)));
    }

    #[test]
    fn test_game_state_new() {
        let state = GameState::new(2, 42);

        assert_eq!(state.player_count(), 2);
        assert_eq!(state.deck_size(PlayerId::new(0)), 0);
        assert_eq!(state.hand(PlayerId::new(0)).len(), 0);
    }

    #[test]
    fn test_game_state_deck_and_draw() {
        let mut state = GameState::new(2, 42);

        state.set_deck(PlayerId::new(0), vec![CardId::new(1), CardId::new(2), CardId::new(3)]);

        let drawn = state.draw_card(PlayerId::new(0));
        assert_eq!(drawn, Some(CardId::new(3))); // Draw from top (end)
        assert_eq!(state.hand(PlayerId::new(0)), &[CardId::new(3)]);
        assert_eq!(state.public.hand_sizes[PlayerId::new(0)], 1);
        assert_eq!(state.deck_size(PlayerId::new(0)), 2);
    }

    #[test]
    fn test_game_state_remove_from_hand() {
        let mut state = GameState::new(2, 42);

        state.add_to_hand(PlayerId::new(0), CardId::new(1));
        state.add_to_hand(PlayerId::new(0), CardId::new(2));

        assert!(state.remove_from_hand(PlayerId::new(0), CardId::new(1)));
        assert_eq!(state.hand(PlayerId::new(0)), &[CardId::new(2)]);
        assert_eq!(state.public.hand_sizes[PlayerId::new(0)], 1);

        assert!(!state.remove_from_hand(PlayerId::new(0), CardId::new(99)));
    }

    #[test]
    fn test_game_state_alloc_entity() {
        let mut state = GameState::new(4, 42);

        let e1 = state.alloc_entity();
        let e2 = state.alloc_entity();

        assert_eq!(e1.0, 4); // First non-player in 4-player game
        assert_eq!(e2.0, 5);
    }

    #[test]
    fn test_game_state_clone() {
        let mut state = GameState::new(2, 42);
        state.set_deck(PlayerId::new(0), vec![CardId::new(1), CardId::new(2)]);
        state.draw_card(PlayerId::new(0));

        let cloned = state.clone_state();

        assert_eq!(cloned.hand(PlayerId::new(0)), state.hand(PlayerId::new(0)));
        assert_eq!(cloned.deck_size(PlayerId::new(0)), state.deck_size(PlayerId::new(0)));
    }

    #[test]
    fn test_four_player_state() {
        let mut state = GameState::new(4, 42);

        let p2 = PlayerId::new(2);
        let p3 = PlayerId::new(3);

        state.set_deck(p2, vec![CardId::new(1), CardId::new(2)]);
        state.set_deck(p3, vec![CardId::new(3), CardId::new(4)]);

        state.draw_card(p2);
        state.draw_card(p3);

        assert_eq!(state.hand(p2), &[CardId::new(2)]);
        assert_eq!(state.hand(p3), &[CardId::new(4)]);
    }
}

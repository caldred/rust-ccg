//! Simple game implementation.

use crate::cards::{CardDefinition, CardId, CardInstance, CardRegistry, CardTypeId};
use crate::core::{
    Action, ActionRecord, EntityId, GameConfig, GameState, PhaseId, PlayerId, TemplateConfig, TemplateId,
    ZoneConfig, ZoneId, ZoneVisibility,
};
use crate::rules::{GameResult, RulesEngine};
use crate::zones::ZonePosition;

/// Action templates for the simple game.
#[derive(Clone, Copy, Debug)]
pub struct Templates {
    /// Draw a card from deck.
    pub draw: TemplateId,
    /// Play a card to attack an opponent.
    pub play: TemplateId,
    /// Pass turn.
    pub pass: TemplateId,
}

impl Templates {
    fn new() -> Self {
        Self {
            draw: TemplateId::new(0),
            play: TemplateId::new(1),
            pass: TemplateId::new(2),
        }
    }
}

/// Zone IDs for the simple game.
/// Each player has their own deck, hand, and discard.
#[derive(Clone, Debug)]
pub struct PlayerZones {
    pub deck: ZoneId,
    pub hand: ZoneId,
    pub discard: ZoneId,
}

/// Simple game state.
#[derive(Clone)]
pub struct SimpleGame {
    config: GameConfig,
    registry: CardRegistry,
    templates: Templates,
    /// Zone IDs per player.
    player_zones: Vec<PlayerZones>,
}

/// Builder for creating a SimpleGame.
pub struct SimpleGameBuilder {
    player_count: usize,
    starting_life: i64,
    cards_per_player: usize,
    starting_hand_size: usize,
}

impl Default for SimpleGameBuilder {
    fn default() -> Self {
        Self {
            player_count: 2,
            starting_life: 20,
            cards_per_player: 10,
            starting_hand_size: 3,
        }
    }
}

impl SimpleGameBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn player_count(mut self, count: usize) -> Self {
        assert!((2..=8).contains(&count), "Player count must be 2-8");
        self.player_count = count;
        self
    }

    pub fn starting_life(mut self, life: i64) -> Self {
        self.starting_life = life;
        self
    }

    pub fn cards_per_player(mut self, count: usize) -> Self {
        self.cards_per_player = count;
        self
    }

    pub fn starting_hand_size(mut self, size: usize) -> Self {
        self.starting_hand_size = size;
        self
    }

    /// Build the game and initial state.
    pub fn build(self, seed: u64) -> (SimpleGame, GameState) {
        let templates = Templates::new();

        // Create zone configs for each player
        let mut zone_configs = Vec::new();
        let mut player_zones = Vec::new();
        let mut zone_id_counter = 0u16;

        for player_idx in 0..self.player_count {
            let player = PlayerId::new(player_idx as u8);

            let deck_id = ZoneId::new(zone_id_counter);
            zone_id_counter += 1;
            let hand_id = ZoneId::new(zone_id_counter);
            zone_id_counter += 1;
            let discard_id = ZoneId::new(zone_id_counter);
            zone_id_counter += 1;

            zone_configs.push(ZoneConfig {
                id: deck_id,
                name: format!("Player {} Deck", player_idx),
                owner: Some(player),
                visibility: ZoneVisibility::Hidden,
                ordered: true,
                max_cards: None,
            });

            zone_configs.push(ZoneConfig {
                id: hand_id,
                name: format!("Player {} Hand", player_idx),
                owner: Some(player),
                visibility: ZoneVisibility::OwnerOnly,
                ordered: false,
                max_cards: None,
            });

            zone_configs.push(ZoneConfig {
                id: discard_id,
                name: format!("Player {} Discard", player_idx),
                owner: Some(player),
                visibility: ZoneVisibility::Public,
                ordered: true,
                max_cards: None,
            });

            player_zones.push(PlayerZones {
                deck: deck_id,
                hand: hand_id,
                discard: discard_id,
            });
        }

        // Create template configs
        let template_configs = vec![
            TemplateConfig {
                id: templates.draw,
                name: "Draw".to_string(),
                pointer_count: 0,
                variable_pointers: false,
            },
            TemplateConfig {
                id: templates.play,
                name: "Play".to_string(),
                pointer_count: 2, // card to play, target opponent
                variable_pointers: false,
            },
            TemplateConfig {
                id: templates.pass,
                name: "Pass".to_string(),
                pointer_count: 0,
                variable_pointers: false,
            },
        ];

        let config = GameConfig {
            player_count: self.player_count,
            zones: zone_configs,
            templates: template_configs,
            initial_phase: PhaseId::default(),
        };

        // Create card registry with simple cards
        let mut registry = CardRegistry::new();
        let card_type = CardTypeId::new(0); // "Creature" type

        // Create card definitions with varying power (1-5)
        for power in 1..=5i64 {
            let card_id = CardId::new(power as u32);
            let def = CardDefinition::new(card_id, format!("Power {} Card", power), card_type)
                .with_attr("power", power);
            registry.register(def);
        }

        // Create game state
        let mut state = GameState::new(self.player_count, seed);

        // Initialize zones
        for pz in &player_zones {
            state.zones.init_ordered_zone(pz.deck);
            state.zones.init_ordered_zone(pz.discard);
        }

        // Set up initial player state (life)
        for player_idx in 0..self.player_count {
            let player = PlayerId::new(player_idx as u8);
            state.public.set_player_state(player, "life", self.starting_life);
        }

        // Create and shuffle decks for each player
        for (player_idx, pz) in player_zones.iter().enumerate().take(self.player_count) {
            let player = PlayerId::new(player_idx as u8);

            // Create cards for this player's deck
            for i in 0..self.cards_per_player {
                let card_id = CardId::new((i % 5 + 1) as u32); // Cycle through power 1-5
                let entity_id = state.alloc_entity();

                let card = CardInstance::new(entity_id, card_id, player, pz.deck);
                // Note: add_card calls zones.add_to_zone internally
                state.add_card(card);
            }

            // Shuffle the deck
            state.zones.shuffle_zone(pz.deck, &mut state.rng);
        }

        // Draw starting hands
        for (player_idx, pz) in player_zones.iter().enumerate().take(self.player_count) {
            let player = PlayerId::new(player_idx as u8);

            for _ in 0..self.starting_hand_size {
                if let Some(entity_id) = state.zones.pop_top(pz.deck) {
                    state.zones.add_to_zone(entity_id, pz.hand, None);
                    if let Some(card) = state.get_card_mut(entity_id) {
                        card.zone = pz.hand;
                    }
                    state.public.hand_sizes[player] += 1;
                }
            }
        }

        let game = SimpleGame {
            config,
            registry,
            templates,
            player_zones,
        };

        (game, state)
    }
}

impl SimpleGame {
    /// Get the templates for this game.
    pub fn templates(&self) -> &Templates {
        &self.templates
    }

    /// Get the zones for a player.
    pub fn player_zones(&self, player: PlayerId) -> &PlayerZones {
        &self.player_zones[player.0 as usize]
    }

    /// Get the card registry.
    pub fn registry(&self) -> &CardRegistry {
        &self.registry
    }

    /// Get a card's power value.
    pub fn card_power(&self, card_id: CardId) -> i64 {
        self.registry
            .get(card_id)
            .map(|def| def.get_int("power", 0))
            .unwrap_or(0)
    }

    /// Check if a player is alive.
    pub fn is_alive(&self, state: &GameState, player: PlayerId) -> bool {
        state.public.get_player_state(player, "life", 0) > 0
    }

    /// Get alive players.
    pub fn alive_players(&self, state: &GameState) -> Vec<PlayerId> {
        PlayerId::all(state.player_count())
            .filter(|&p| self.is_alive(state, p))
            .collect()
    }

    /// Get opponents of a player (other alive players).
    pub fn opponents(&self, state: &GameState, player: PlayerId) -> Vec<PlayerId> {
        self.alive_players(state)
            .into_iter()
            .filter(|&p| p != player)
            .collect()
    }
}

impl RulesEngine for SimpleGame {
    fn config(&self) -> &GameConfig {
        &self.config
    }

    fn legal_templates(&self, state: &GameState, player: PlayerId) -> Vec<TemplateId> {
        // Only active player can act
        if state.public.active_player != player || !self.is_alive(state, player) {
            return vec![];
        }

        let mut templates = Vec::new();
        let pz = self.player_zones(player);

        // Can draw if deck has cards
        if state.zones.zone_size(pz.deck) > 0 {
            templates.push(self.templates.draw);
        }

        // Can play if has cards in hand and has opponents
        if state.public.hand_sizes[player] > 0 && !self.opponents(state, player).is_empty() {
            templates.push(self.templates.play);
        }

        // Can always pass
        templates.push(self.templates.pass);

        templates
    }

    fn legal_pointers(
        &self,
        state: &GameState,
        player: PlayerId,
        template: TemplateId,
        prior_pointers: &[EntityId],
    ) -> Vec<EntityId> {
        if template == self.templates.play {
            let pz = self.player_zones(player);

            match prior_pointers.len() {
                0 => {
                    // First pointer: card in hand
                    state.zones
                        .cards_in_zone(pz.hand)
                        .collect()
                }
                1 => {
                    // Second pointer: target opponent (as EntityId)
                    self.opponents(state, player)
                        .into_iter()
                        .map(EntityId::player)
                        .collect()
                }
                _ => vec![], // No more pointers needed
            }
        } else {
            // Draw and Pass have no pointers
            vec![]
        }
    }

    fn apply_action(&mut self, state: &mut GameState, player: PlayerId, action: &Action) {
        let templates = self.templates;

        if action.template == templates.draw {
            // Draw a card
            let pz = &self.player_zones[player.0 as usize];
            if let Some(entity_id) = state.zones.pop_top(pz.deck) {
                state.zones.add_to_zone(entity_id, pz.hand, None);
                if let Some(card) = state.get_card_mut(entity_id) {
                    card.zone = pz.hand;
                }
                state.public.hand_sizes[player] += 1;
            }
        } else if action.template == templates.play {
            // Play a card to deal damage
            let card_entity = action.pointers[0];
            let target_entity = action.pointers[1];
            let target_player = target_entity
                .as_player(state.player_count())
                .expect("Target must be a player");

            // Get card power
            let power = if let Some(card) = state.get_card(card_entity) {
                self.card_power(card.card_id)
            } else {
                0
            };

            // Discard the card
            let pz = &self.player_zones[player.0 as usize];
            state.zones.move_to_zone(card_entity, pz.discard, Some(ZonePosition::Top));
            if let Some(card) = state.get_card_mut(card_entity) {
                card.zone = pz.discard;
            }
            state.public.hand_sizes[player] -= 1;

            // Deal damage
            state.public.modify_player_state(target_player, "life", -power);
        }
        // Pass: do nothing

        // Advance to next player
        let alive = self.alive_players(state);
        if alive.len() > 1 {
            // Find next alive player
            let current_idx = alive.iter().position(|&p| p == player).unwrap_or(0);
            let next_idx = (current_idx + 1) % alive.len();
            let next_player = alive[next_idx];

            state.public.set_active_player(next_player);
            state.public.set_priority(next_player);
        }

        // Record action
        let seq = state.public.next_sequence();
        state.public.record_action(ActionRecord {
            player,
            action: action.clone(),
            turn: state.public.turn_number,
            sequence: seq,
        });
    }

    fn is_terminal(&self, state: &GameState) -> Option<GameResult> {
        let alive = self.alive_players(state);

        match alive.len() {
            0 => Some(GameResult::Draw), // Everyone dead (shouldn't happen normally)
            1 => Some(GameResult::Winner(alive[0])),
            _ => None, // Game continues
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_game_creation() {
        let (game, state) = SimpleGameBuilder::new()
            .player_count(2)
            .starting_life(20)
            .cards_per_player(10)
            .starting_hand_size(3)
            .build(42);

        assert_eq!(state.player_count(), 2);
        assert_eq!(state.public.get_player_state(PlayerId::new(0), "life", 0), 20);
        assert_eq!(state.public.get_player_state(PlayerId::new(1), "life", 0), 20);
        assert_eq!(state.public.hand_sizes[PlayerId::new(0)], 3);
        assert_eq!(state.public.hand_sizes[PlayerId::new(1)], 3);

        // Check deck sizes (10 - 3 starting hand = 7)
        let p0_zones = game.player_zones(PlayerId::new(0));
        assert_eq!(state.zones.zone_size(p0_zones.deck), 7);
    }

    #[test]
    fn test_four_player_game() {
        let (game, state) = SimpleGameBuilder::new()
            .player_count(4)
            .starting_life(15)
            .build(42);

        assert_eq!(state.player_count(), 4);

        for i in 0..4 {
            let player = PlayerId::new(i);
            assert_eq!(state.public.get_player_state(player, "life", 0), 15);
            assert!(game.is_alive(&state, player));
        }

        assert_eq!(game.alive_players(&state).len(), 4);
        assert_eq!(game.opponents(&state, PlayerId::new(0)).len(), 3);
    }

    #[test]
    fn test_legal_actions() {
        let (game, state) = SimpleGameBuilder::new()
            .player_count(2)
            .build(42);

        let player = PlayerId::new(0);
        let templates = game.legal_templates(&state, player);

        // Should have Draw, Play, and Pass
        assert!(templates.contains(&game.templates().draw));
        assert!(templates.contains(&game.templates().play));
        assert!(templates.contains(&game.templates().pass));
    }

    #[test]
    fn test_play_action() {
        let (mut game, mut state) = SimpleGameBuilder::new()
            .player_count(2)
            .starting_life(20)
            .build(42);

        let player = PlayerId::new(0);
        let opponent = PlayerId::new(1);

        // Get a card from hand
        let pz = game.player_zones(player);
        let cards_in_hand: Vec<_> = state.zones.cards_in_zone(pz.hand).collect();
        assert!(!cards_in_hand.is_empty());

        let card_entity = cards_in_hand[0];
        let card_power = state
            .get_card(card_entity)
            .map(|c| game.card_power(c.card_id))
            .unwrap_or(0);

        // Create and apply play action
        let action = Action::with_pointers(
            game.templates().play,
            &[card_entity, EntityId::player(opponent)],
        );

        game.apply_action(&mut state, player, &action);

        // Check damage was dealt
        let opponent_life = state.public.get_player_state(opponent, "life", 0);
        assert_eq!(opponent_life, 20 - card_power);

        // Check card was discarded
        assert_eq!(state.public.hand_sizes[player], 2); // Started with 3, played 1

        // Check turn passed
        assert_eq!(state.public.active_player, opponent);
    }

    #[test]
    fn test_draw_action() {
        let (mut game, mut state) = SimpleGameBuilder::new()
            .player_count(2)
            .build(42);

        let player = PlayerId::new(0);
        let initial_hand = state.public.hand_sizes[player];
        let deck_zone = game.player_zones(player).deck;
        let initial_deck = state.zones.zone_size(deck_zone);

        let action = Action::new(game.templates().draw);
        game.apply_action(&mut state, player, &action);

        assert_eq!(state.public.hand_sizes[player], initial_hand + 1);
        assert_eq!(state.zones.zone_size(deck_zone), initial_deck - 1);
    }

    #[test]
    fn test_game_to_completion() {
        let (mut game, mut state) = SimpleGameBuilder::new()
            .player_count(2)
            .starting_life(5) // Low life for quick game
            .cards_per_player(20)
            .build(42);

        let mut turn_count = 0;
        const MAX_TURNS: usize = 1000;

        while game.is_terminal(&state).is_none() && turn_count < MAX_TURNS {
            let active = state.public.active_player;
            let actions = game.legal_actions(&state, active);

            if actions.is_empty() {
                break;
            }

            // Simple AI: prefer playing cards, then draw, then pass
            let action = actions
                .iter()
                .find(|a| a.template == game.templates().play)
                .or_else(|| actions.iter().find(|a| a.template == game.templates().draw))
                .or_else(|| actions.iter().find(|a| a.template == game.templates().pass))
                .cloned()
                .unwrap();

            game.apply_action(&mut state, active, &action);
            turn_count += 1;
        }

        let result = game.is_terminal(&state);
        assert!(result.is_some(), "Game should have ended");

        match result.unwrap() {
            GameResult::Winner(winner) => {
                assert!(game.is_alive(&state, winner));
            }
            GameResult::Draw => {
                // Both died at the same time, which is valid
            }
            _ => panic!("Unexpected result"),
        }
    }

    #[test]
    fn test_deterministic_replay() {
        let seed = 12345u64;

        // Play game twice with same seed
        let (mut game1, mut state1) = SimpleGameBuilder::new()
            .player_count(2)
            .starting_life(10)
            .build(seed);

        let (mut game2, mut state2) = SimpleGameBuilder::new()
            .player_count(2)
            .starting_life(10)
            .build(seed);

        // Record first game's actions
        let mut actions_taken = Vec::new();
        let mut turn = 0;

        while game1.is_terminal(&state1).is_none() && turn < 100 {
            let active = state1.public.active_player;
            let actions = game1.legal_actions(&state1, active);

            if actions.is_empty() {
                break;
            }

            // Pick first action consistently
            let action = actions[0].clone();
            actions_taken.push((active, action.clone()));

            game1.apply_action(&mut state1, active, &action);
            turn += 1;
        }

        // Replay on second game
        for (player, action) in &actions_taken {
            game2.apply_action(&mut state2, *player, action);
        }

        // States should be identical
        assert_eq!(
            state1.public.get_player_state(PlayerId::new(0), "life", 0),
            state2.public.get_player_state(PlayerId::new(0), "life", 0)
        );
        assert_eq!(
            state1.public.get_player_state(PlayerId::new(1), "life", 0),
            state2.public.get_player_state(PlayerId::new(1), "life", 0)
        );
        assert_eq!(game1.is_terminal(&state1), game2.is_terminal(&state2));
    }

    #[test]
    fn test_six_player_game() {
        let (mut game, mut state) = SimpleGameBuilder::new()
            .player_count(6)
            .starting_life(5)
            .build(42);

        // Verify 6 players work
        assert_eq!(state.player_count(), 6);
        assert_eq!(game.alive_players(&state).len(), 6);

        // Play until one player dies
        let mut turns = 0;
        while game.alive_players(&state).len() == 6 && turns < 200 {
            let active = state.public.active_player;
            let actions = game.legal_actions(&state, active);

            if actions.is_empty() {
                break;
            }

            // Prefer attacking
            let action = actions
                .iter()
                .find(|a| a.template == game.templates().play)
                .cloned()
                .unwrap_or_else(|| actions[0].clone());

            game.apply_action(&mut state, active, &action);
            turns += 1;
        }

        // Should have at least one death or timeout
        assert!(turns > 0);
    }
}

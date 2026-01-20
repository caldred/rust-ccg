//! Effect resolution - executing effects on game state.
//!
//! The `EffectResolver` applies effects to game state in a
//! game-agnostic way. Games can provide callbacks for custom
//! behavior (like determining deck/hand zones for draw effects).

use crate::core::{EntityId, GameState, PlayerId, ZoneId};

use super::{Effect, EffectBatch};

/// Context for resolving effects.
///
/// Games provide this to give the resolver access to game-specific
/// information like zone mappings for draw effects.
pub struct ResolverContext<'a> {
    /// Get the deck zone for a player.
    pub get_deck_zone: Box<dyn Fn(PlayerId) -> ZoneId + 'a>,
    /// Get the hand zone for a player.
    pub get_hand_zone: Box<dyn Fn(PlayerId) -> ZoneId + 'a>,
    /// Evaluate a custom condition.
    pub eval_condition: Box<dyn Fn(&str, &GameState) -> bool + 'a>,
}

impl<'a> ResolverContext<'a> {
    /// Create a new context with required zone mappings.
    pub fn new(
        get_deck_zone: impl Fn(PlayerId) -> ZoneId + 'a,
        get_hand_zone: impl Fn(PlayerId) -> ZoneId + 'a,
    ) -> Self {
        Self {
            get_deck_zone: Box::new(get_deck_zone),
            get_hand_zone: Box::new(get_hand_zone),
            eval_condition: Box::new(|_, _| false),
        }
    }

    /// Add a condition evaluator.
    pub fn with_condition_eval(
        mut self,
        eval: impl Fn(&str, &GameState) -> bool + 'a,
    ) -> Self {
        self.eval_condition = Box::new(eval);
        self
    }
}

/// Result of resolving an effect.
#[derive(Clone, Debug)]
pub enum ResolveResult {
    /// Effect resolved successfully.
    Success,
    /// Effect failed (e.g., invalid target).
    Failed(String),
    /// Effect was skipped (e.g., conditional not met).
    Skipped,
}

/// Resolves effects on game state.
pub struct EffectResolver;

impl EffectResolver {
    /// Resolve a batch of effects.
    pub fn resolve_batch(
        state: &mut GameState,
        batch: &EffectBatch,
        context: &ResolverContext,
    ) -> Vec<ResolveResult> {
        let mut results = Vec::new();

        for entry in batch.iter() {
            for target in &entry.targets {
                let result = Self::resolve_single(state, &entry.effect, *target, context);
                results.push(result);
            }

            // Effects without targets (like ShuffleZone)
            if entry.targets.is_empty() {
                let result = Self::resolve_zone_effect(state, &entry.effect, context);
                results.push(result);
            }
        }

        results
    }

    /// Resolve a single effect on a target.
    pub fn resolve_single(
        state: &mut GameState,
        effect: &Effect,
        target: EntityId,
        context: &ResolverContext,
    ) -> ResolveResult {
        let player_count = state.player_count();

        match effect {
            Effect::ModifyPlayerState { key, delta } => {
                if let Some(idx) = target.as_player_index(player_count) {
                    let player = PlayerId::new(idx);
                    state.public.modify_player_state(player, key, *delta);
                    ResolveResult::Success
                } else {
                    ResolveResult::Failed("Target is not a player".to_string())
                }
            }

            Effect::SetPlayerState { key, value } => {
                if let Some(idx) = target.as_player_index(player_count) {
                    let player = PlayerId::new(idx);
                    state.public.set_player_state(player, key, *value);
                    ResolveResult::Success
                } else {
                    ResolveResult::Failed("Target is not a player".to_string())
                }
            }

            Effect::MoveCard { destination, position } => {
                if !target.is_player(player_count) {
                    state.zones.move_to_zone(target, *destination, *position);
                    if let Some(card) = state.get_card_mut(target) {
                        card.zone = *destination;
                    }
                    ResolveResult::Success
                } else {
                    ResolveResult::Failed("Target is a player, not a card".to_string())
                }
            }

            Effect::DrawCards { count, from_zone, to_zone } => {
                if let Some(idx) = target.as_player_index(player_count) {
                    let player = PlayerId::new(idx);
                    let deck = from_zone.unwrap_or_else(|| (context.get_deck_zone)(player));
                    let hand = to_zone.unwrap_or_else(|| (context.get_hand_zone)(player));

                    let mut drawn = 0;
                    for _ in 0..*count {
                        if let Some(entity_id) = state.zones.pop_top(deck) {
                            state.zones.add_to_zone(entity_id, hand, None);
                            if let Some(card) = state.get_card_mut(entity_id) {
                                card.zone = hand;
                            }
                            state.public.hand_sizes[player] += 1;
                            drawn += 1;
                        } else {
                            break;
                        }
                    }

                    if drawn > 0 {
                        ResolveResult::Success
                    } else {
                        ResolveResult::Failed("Deck was empty".to_string())
                    }
                } else {
                    ResolveResult::Failed("Target is not a player".to_string())
                }
            }

            Effect::ModifyCardState { key, delta } => {
                if !target.is_player(player_count) {
                    if let Some(card) = state.get_card_mut(target) {
                        card.modify_state(key, *delta);
                        ResolveResult::Success
                    } else {
                        ResolveResult::Failed("Card not found".to_string())
                    }
                } else {
                    ResolveResult::Failed("Target is a player, not a card".to_string())
                }
            }

            Effect::SetCardState { key, value } => {
                if !target.is_player(player_count) {
                    if let Some(card) = state.get_card_mut(target) {
                        card.set_state(key, *value);
                        ResolveResult::Success
                    } else {
                        ResolveResult::Failed("Card not found".to_string())
                    }
                } else {
                    ResolveResult::Failed("Target is a player, not a card".to_string())
                }
            }

            Effect::ModifyTurnState { key, delta } => {
                let current = state.public.get_turn_state(key, 0);
                state.public.set_turn_state(key, current + delta);
                ResolveResult::Success
            }

            Effect::SetTurnState { key, value } => {
                state.public.set_turn_state(key, *value);
                ResolveResult::Success
            }

            Effect::ShuffleZone { zone } => {
                // This is a zone effect, not a target effect
                state.zones.shuffle_zone(*zone, &mut state.rng);
                ResolveResult::Success
            }

            Effect::Batch(effects) => {
                for sub_effect in effects {
                    let result = Self::resolve_single(state, sub_effect, target, context);
                    if matches!(result, ResolveResult::Failed(_)) {
                        return result;
                    }
                }
                ResolveResult::Success
            }

            Effect::Conditional { condition_key, effect } => {
                if (context.eval_condition)(condition_key, state) {
                    Self::resolve_single(state, effect, target, context)
                } else {
                    ResolveResult::Skipped
                }
            }
        }
    }

    /// Resolve a zone-level effect (no target entity).
    fn resolve_zone_effect(
        state: &mut GameState,
        effect: &Effect,
        _context: &ResolverContext,
    ) -> ResolveResult {
        match effect {
            Effect::ShuffleZone { zone } => {
                state.zones.shuffle_zone(*zone, &mut state.rng);
                ResolveResult::Success
            }
            Effect::ModifyTurnState { key, delta } => {
                let current = state.public.get_turn_state(key, 0);
                state.public.set_turn_state(key, current + delta);
                ResolveResult::Success
            }
            Effect::SetTurnState { key, value } => {
                state.public.set_turn_state(key, *value);
                ResolveResult::Success
            }
            _ => ResolveResult::Failed("Effect requires a target".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::{CardId, CardInstance};

    fn setup_test_state() -> (GameState, ZoneId, ZoneId) {
        let mut state = GameState::new(2, 42);
        let deck = ZoneId::new(0);
        let hand = ZoneId::new(1);

        state.zones.init_ordered_zone(deck);

        // Add some cards to deck
        for i in 0..5 {
            let entity = EntityId(10 + i);
            let card = CardInstance::new(entity, CardId::new(1), PlayerId::new(0), deck);
            state.add_card(card);
        }

        (state, deck, hand)
    }

    fn test_context(deck: ZoneId, hand: ZoneId) -> ResolverContext<'static> {
        ResolverContext::new(
            move |_| deck,
            move |_| hand,
        )
    }

    #[test]
    fn test_damage_effect() {
        let mut state = GameState::new(2, 42);
        state.public.set_player_state(PlayerId::new(1), "life", 20);

        let context = ResolverContext::new(|_| ZoneId::new(0), |_| ZoneId::new(1));
        let effect = Effect::damage(5);
        let target = EntityId::player_id(1);

        let result = EffectResolver::resolve_single(&mut state, &effect, target, &context);

        assert!(matches!(result, ResolveResult::Success));
        assert_eq!(state.public.get_player_state(PlayerId::new(1), "life", 0), 15);
    }

    #[test]
    fn test_heal_effect() {
        let mut state = GameState::new(2, 42);
        state.public.set_player_state(PlayerId::new(0), "life", 10);

        let context = ResolverContext::new(|_| ZoneId::new(0), |_| ZoneId::new(1));
        let effect = Effect::heal(5);
        let target = EntityId::player_id(0);

        EffectResolver::resolve_single(&mut state, &effect, target, &context);

        assert_eq!(state.public.get_player_state(PlayerId::new(0), "life", 0), 15);
    }

    #[test]
    fn test_draw_effect() {
        let (mut state, deck, hand) = setup_test_state();
        let context = test_context(deck, hand);

        let effect = Effect::draw(2);
        let target = EntityId::player_id(0);

        let result = EffectResolver::resolve_single(&mut state, &effect, target, &context);

        assert!(matches!(result, ResolveResult::Success));
        assert_eq!(state.zones.zone_size(deck), 3); // 5 - 2 = 3
        assert_eq!(state.public.hand_sizes[PlayerId::new(0)], 2);
    }

    #[test]
    fn test_draw_empty_deck() {
        let mut state = GameState::new(2, 42);
        let deck = ZoneId::new(0);
        let hand = ZoneId::new(1);
        state.zones.init_ordered_zone(deck);

        let context = test_context(deck, hand);
        let effect = Effect::draw(1);
        let target = EntityId::player_id(0);

        let result = EffectResolver::resolve_single(&mut state, &effect, target, &context);

        assert!(matches!(result, ResolveResult::Failed(_)));
    }

    #[test]
    fn test_move_card_effect() {
        let (mut state, deck, hand) = setup_test_state();
        let context = test_context(deck, hand);

        let card_entity = EntityId(10);
        let discard = ZoneId::new(2);
        state.zones.init_ordered_zone(discard);

        let effect = Effect::move_to_top(discard);

        let result = EffectResolver::resolve_single(&mut state, &effect, card_entity, &context);

        assert!(matches!(result, ResolveResult::Success));
        assert!(state.zones.is_in_zone(card_entity, discard));
    }

    #[test]
    fn test_modify_card_state() {
        let (mut state, deck, hand) = setup_test_state();
        let context = test_context(deck, hand);

        let card_entity = EntityId(10);
        let effect = Effect::modify_card("damage", 3);

        EffectResolver::resolve_single(&mut state, &effect, card_entity, &context);

        let card = state.get_card(card_entity).unwrap();
        assert_eq!(card.get_state("damage", 0), 3);
    }

    #[test]
    fn test_batch_effect() {
        let mut state = GameState::new(2, 42);
        state.public.set_player_state(PlayerId::new(0), "life", 20);

        let context = ResolverContext::new(|_| ZoneId::new(0), |_| ZoneId::new(1));
        let effect = Effect::batch([
            Effect::damage(3),
            Effect::modify_player("mana", 2),
        ]);
        let target = EntityId::player_id(0);

        EffectResolver::resolve_single(&mut state, &effect, target, &context);

        assert_eq!(state.public.get_player_state(PlayerId::new(0), "life", 0), 17);
        assert_eq!(state.public.get_player_state(PlayerId::new(0), "mana", 0), 2);
    }

    #[test]
    fn test_effect_batch_resolution() {
        let mut state = GameState::new(2, 42);
        state.public.set_player_state(PlayerId::new(0), "life", 20);
        state.public.set_player_state(PlayerId::new(1), "life", 20);

        let context = ResolverContext::new(|_| ZoneId::new(0), |_| ZoneId::new(1));

        let mut batch = EffectBatch::new();
        batch.add_player(Effect::damage(3), PlayerId::new(0));
        batch.add_player(Effect::damage(5), PlayerId::new(1));

        let results = EffectResolver::resolve_batch(&mut state, &batch, &context);

        assert_eq!(results.len(), 2);
        assert!(matches!(results[0], ResolveResult::Success));
        assert!(matches!(results[1], ResolveResult::Success));

        assert_eq!(state.public.get_player_state(PlayerId::new(0), "life", 0), 17);
        assert_eq!(state.public.get_player_state(PlayerId::new(1), "life", 0), 15);
    }

    #[test]
    fn test_conditional_effect() {
        let mut state = GameState::new(2, 42);
        state.public.set_player_state(PlayerId::new(0), "life", 20);
        state.public.set_turn_state("can_heal", 1);

        let context = ResolverContext::new(|_| ZoneId::new(0), |_| ZoneId::new(1))
            .with_condition_eval(|key, state| {
                state.public.get_turn_state(key, 0) != 0
            });

        let effect = Effect::Conditional {
            condition_key: "can_heal".to_string(),
            effect: Box::new(Effect::heal(5)),
        };

        EffectResolver::resolve_single(&mut state, &effect, EntityId::player_id(0), &context);

        assert_eq!(state.public.get_player_state(PlayerId::new(0), "life", 0), 25);
    }

    #[test]
    fn test_conditional_effect_skipped() {
        let mut state = GameState::new(2, 42);
        state.public.set_player_state(PlayerId::new(0), "life", 20);
        // can_heal is 0 (default)

        let context = ResolverContext::new(|_| ZoneId::new(0), |_| ZoneId::new(1))
            .with_condition_eval(|key, state| {
                state.public.get_turn_state(key, 0) != 0
            });

        let effect = Effect::Conditional {
            condition_key: "can_heal".to_string(),
            effect: Box::new(Effect::heal(5)),
        };

        let result = EffectResolver::resolve_single(&mut state, &effect, EntityId::player_id(0), &context);

        assert!(matches!(result, ResolveResult::Skipped));
        assert_eq!(state.public.get_player_state(PlayerId::new(0), "life", 0), 20); // Unchanged
    }
}

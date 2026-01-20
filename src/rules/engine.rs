//! Rules engine trait for game implementations.
//!
//! Games implement `RulesEngine` to define their rules:
//! - What actions are legal
//! - How actions modify state
//! - Win/loss conditions

use crate::core::action::Action;
use crate::core::config::{GameConfig, TemplateId};
use crate::core::entity::EntityId;
use crate::core::player::PlayerId;
use crate::core::state::GameState;

/// Result of a completed game.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GameResult {
    /// Single winner.
    Winner(PlayerId),
    /// Draw (no winner).
    Draw,
    /// Multiple winners (team games, shared victory).
    Winners(Vec<PlayerId>),
}

impl GameResult {
    /// Check if a player won.
    #[must_use]
    pub fn is_winner(&self, player: PlayerId) -> bool {
        match self {
            GameResult::Winner(p) => *p == player,
            GameResult::Winners(ps) => ps.contains(&player),
            GameResult::Draw => false,
        }
    }
}

/// Rules engine trait.
///
/// Games implement this trait to define their rules.
/// The engine calls these methods during gameplay and MCTS.
///
/// ## Implementation Notes
///
/// - `legal_templates`: Return empty vec if player can't act
/// - `legal_pointers`: Called iteratively for multi-pointer actions
/// - `apply_action`: Must be deterministic for MCTS
/// - `is_terminal`: Return None if game continues
pub trait RulesEngine {
    /// Get the game configuration.
    fn config(&self) -> &GameConfig;

    /// Get legal action templates for a player.
    ///
    /// Returns empty if the player has no legal actions.
    fn legal_templates(&self, state: &GameState, player: PlayerId) -> Vec<TemplateId>;

    /// Get legal entity pointers for an action being built.
    ///
    /// Called iteratively as pointers are selected:
    /// - First call: `prior_pointers` is empty
    /// - Second call: `prior_pointers` has first pointer
    /// - etc.
    ///
    /// Returns empty when no more pointers are needed or none are legal.
    fn legal_pointers(
        &self,
        state: &GameState,
        player: PlayerId,
        template: TemplateId,
        prior_pointers: &[EntityId],
    ) -> Vec<EntityId>;

    /// Apply an action to the game state.
    ///
    /// Must be deterministic for MCTS consistency.
    fn apply_action(&mut self, state: &mut GameState, player: PlayerId, action: &Action);

    /// Check if the game is over.
    ///
    /// Returns `Some(result)` if the game has ended, `None` if it continues.
    fn is_terminal(&self, state: &GameState) -> Option<GameResult>;

    // === Convenience Methods ===

    /// Enumerate all legal actions for a player.
    ///
    /// Default implementation builds actions from templates and pointers.
    fn legal_actions(&self, state: &GameState, player: PlayerId) -> Vec<Action> {
        let mut actions = Vec::new();

        for template in self.legal_templates(state, player) {
            self.enumerate_actions_for_template(state, player, template, &[], &mut actions);
        }

        actions
    }

    /// Helper to enumerate actions for a template recursively.
    fn enumerate_actions_for_template(
        &self,
        state: &GameState,
        player: PlayerId,
        template: TemplateId,
        prior_pointers: &[EntityId],
        out: &mut Vec<Action>,
    ) {
        let next_pointers = self.legal_pointers(state, player, template, prior_pointers);

        if next_pointers.is_empty() {
            // No more pointers needed - this is a complete action
            out.push(Action::with_pointers(template, prior_pointers));
        } else {
            // Recurse for each legal pointer
            for pointer in next_pointers {
                let mut pointers = prior_pointers.to_vec();
                pointers.push(pointer);
                self.enumerate_actions_for_template(state, player, template, &pointers, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_game_result_is_winner() {
        let result = GameResult::Winner(PlayerId::new(1));
        assert!(!result.is_winner(PlayerId::new(0)));
        assert!(result.is_winner(PlayerId::new(1)));

        let draw = GameResult::Draw;
        assert!(!draw.is_winner(PlayerId::new(0)));

        let team = GameResult::Winners(vec![PlayerId::new(0), PlayerId::new(2)]);
        assert!(team.is_winner(PlayerId::new(0)));
        assert!(!team.is_winner(PlayerId::new(1)));
        assert!(team.is_winner(PlayerId::new(2)));
    }
}

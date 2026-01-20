//! Game bindings for Python.

use pyo3::prelude::*;

use crate::core::GameState;
use crate::games::simple::{SimpleGame, SimpleGameBuilder};
use crate::rules::{GameResult, RulesEngine};

use super::py_core::{PyAction, PyPlayerId};

/// Python wrapper for SimpleGame.
///
/// A simple card game for testing the engine.
#[pyclass(name = "SimpleGame")]
pub struct PySimpleGame {
    rules: SimpleGame,
    state: GameState,
}

#[pymethods]
impl PySimpleGame {
    /// Create a new SimpleGame.
    ///
    /// # Arguments
    /// - player_count: Number of players (2-6)
    /// - starting_life: Starting life total for each player
    /// - starting_hand_size: Number of cards to draw at start
    /// - cards_per_player: Number of cards in each player's deck
    /// - seed: RNG seed for deterministic games
    #[new]
    #[pyo3(signature = (
        player_count = 2,
        starting_life = 20,
        starting_hand_size = 5,
        cards_per_player = 10,
        seed = 42
    ))]
    fn new(
        player_count: usize,
        starting_life: i64,
        starting_hand_size: usize,
        cards_per_player: usize,
        seed: u64,
    ) -> Self {
        let (rules, state) = SimpleGameBuilder::new()
            .player_count(player_count)
            .starting_life(starting_life)
            .starting_hand_size(starting_hand_size)
            .cards_per_player(cards_per_player)
            .build(seed);
        Self { rules, state }
    }

    /// Get legal actions for the current player.
    fn legal_actions(&self) -> Vec<PyAction> {
        let player = self.state.public.active_player;
        self.rules
            .legal_actions(&self.state, player)
            .into_iter()
            .map(PyAction)
            .collect()
    }

    /// Apply an action.
    fn apply_action(&mut self, action: &PyAction) {
        let player = self.state.public.active_player;
        self.rules
            .apply_action(&mut self.state, player, &action.0);
    }

    /// Check if the game is terminal.
    ///
    /// Returns the winner if the game is over, None otherwise.
    fn is_terminal(&self) -> Option<PyPlayerId> {
        match self.rules.is_terminal(&self.state) {
            Some(GameResult::Winner(p)) => Some(PyPlayerId(p)),
            _ => None,
        }
    }

    /// Check if a specific player has won.
    fn has_winner(&self) -> bool {
        matches!(self.rules.is_terminal(&self.state), Some(GameResult::Winner(_)))
    }

    /// Get the active player (whose turn it is).
    #[getter]
    fn active_player(&self) -> PyPlayerId {
        PyPlayerId(self.state.public.active_player)
    }

    /// Get the current turn number.
    #[getter]
    fn turn_number(&self) -> u32 {
        self.state.public.turn_number
    }

    /// Get the number of players.
    #[getter]
    fn player_count(&self) -> usize {
        self.state.player_count()
    }

    /// Get a player's life total.
    fn get_life(&self, player: &PyPlayerId) -> i64 {
        self.state.public.get_player_state(player.0, "life", 0)
    }

    /// Get a player's hand size.
    fn get_hand_size(&self, player: &PyPlayerId) -> u32 {
        self.state.public.hand_sizes[player.0]
    }

    /// Copy the game state for simulation.
    fn copy(&mut self) -> Self {
        Self {
            rules: self.rules.clone(),
            state: self.state.clone_state(),
        }
    }

    fn __repr__(&self) -> String {
        let terminal = if self.is_terminal().is_some() {
            "terminal"
        } else {
            "ongoing"
        };
        format!(
            "SimpleGame(turn={}, active=P{}, status={})",
            self.state.public.turn_number, self.state.public.active_player.0, terminal
        )
    }
}

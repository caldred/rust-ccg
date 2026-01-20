//! Self-play bindings for Python.

use pyo3::prelude::*;

use crate::games::simple::{SimpleGame, SimpleGameBuilder};
use crate::nn::SimpleGameEncoder;
use crate::training::{SelfPlayConfig, SelfPlayWorker};

use super::py_nn::PyPolicyValueNetwork;
use super::py_training::PyTrajectory;

/// Python wrapper for SelfPlayConfig.
#[pyclass(name = "SelfPlayConfig")]
#[derive(Clone)]
pub struct PySelfPlayConfig(pub SelfPlayConfig);

#[pymethods]
impl PySelfPlayConfig {
    /// Create a new self-play configuration.
    ///
    /// # Arguments
    /// - mcts_iterations: Number of MCTS iterations per move (default: 800)
    /// - temperature: Exploration temperature for action selection (default: 1.0)
    /// - temperature_threshold: Move number to switch to greedy selection (default: 30)
    /// - max_moves: Maximum moves before declaring a draw (default: 500)
    /// - exploration_constant: MCTS exploration constant (default: 1.414)
    #[new]
    #[pyo3(signature = (
        mcts_iterations = 800,
        temperature = 1.0,
        temperature_threshold = 30,
        max_moves = 500,
        exploration_constant = 1.414
    ))]
    fn new(
        mcts_iterations: u32,
        temperature: f64,
        temperature_threshold: usize,
        max_moves: usize,
        exploration_constant: f64,
    ) -> Self {
        Self(
            SelfPlayConfig::default()
                .with_mcts_iterations(mcts_iterations)
                .with_temperature(temperature)
                .with_temperature_threshold(temperature_threshold)
                .with_max_moves(max_moves)
                .with_exploration(exploration_constant),
        )
    }

    #[getter]
    fn mcts_iterations(&self) -> u32 {
        self.0.mcts_iterations
    }

    #[getter]
    fn temperature(&self) -> f64 {
        self.0.temperature
    }

    #[getter]
    fn temperature_threshold(&self) -> usize {
        self.0.temperature_threshold
    }

    #[getter]
    fn max_moves(&self) -> usize {
        self.0.max_moves
    }

    #[getter]
    fn exploration_constant(&self) -> f64 {
        self.0.exploration_constant
    }

    fn __repr__(&self) -> String {
        format!(
            "SelfPlayConfig(iters={}, temp={}, threshold={}, max_moves={})",
            self.0.mcts_iterations,
            self.0.temperature,
            self.0.temperature_threshold,
            self.0.max_moves
        )
    }
}

/// Self-play worker for SimpleGame.
///
/// This is a concrete implementation that avoids generic type issues with PyO3.
#[pyclass(name = "SimpleGameWorker")]
pub struct PySimpleGameWorker {
    inner: SelfPlayWorker<SimpleGame>,
    player_count: usize,
    starting_life: i64,
    starting_hand_size: usize,
    cards_per_player: usize,
}

impl PySimpleGameWorker {
    /// Create a new game state with the configured parameters.
    fn create_game_state(&self, seed: u64) -> crate::core::GameState {
        let (_rules, state) = SimpleGameBuilder::new()
            .player_count(self.player_count)
            .starting_life(self.starting_life)
            .starting_hand_size(self.starting_hand_size)
            .cards_per_player(self.cards_per_player)
            .build(seed);
        state
    }
}

#[pymethods]
impl PySimpleGameWorker {
    /// Create a new self-play worker for SimpleGame.
    ///
    /// # Arguments
    /// - player_count: Number of players (2-6)
    /// - config: Self-play configuration
    /// - starting_life: Starting life for each player (default: 20)
    /// - starting_hand_size: Cards in starting hand (default: 5)
    /// - cards_per_player: Cards in each deck (default: 10)
    #[new]
    #[pyo3(signature = (
        player_count,
        config,
        starting_life = 20,
        starting_hand_size = 5,
        cards_per_player = 10
    ))]
    fn new(
        player_count: usize,
        config: &PySelfPlayConfig,
        starting_life: i64,
        starting_hand_size: usize,
        cards_per_player: usize,
    ) -> Self {
        let (rules, _state) = SimpleGameBuilder::new()
            .player_count(player_count)
            .starting_life(starting_life)
            .starting_hand_size(starting_hand_size)
            .cards_per_player(cards_per_player)
            .build(0);

        // Action space: pass (0), play card 0-9 (1-10) = 11 actions
        // This matches the SimpleGame implementation
        let action_space = 1 + cards_per_player;
        let encoder = Box::new(SimpleGameEncoder::new(player_count, action_space));

        Self {
            inner: SelfPlayWorker::new(rules, encoder, config.0.clone()),
            player_count,
            starting_life,
            starting_hand_size,
            cards_per_player,
        }
    }

    /// Play a game using pure MCTS (no neural network).
    ///
    /// Returns a trajectory containing all moves and the final outcome.
    fn play_game(&self, seed: u64) -> PyTrajectory {
        let mut state = self.create_game_state(seed);
        PyTrajectory(self.inner.play_game(&mut state, seed))
    }

    /// Play a game using MCTS guided by a neural network.
    ///
    /// The network provides policy priors and value estimates.
    fn play_game_with_network(&self, seed: u64, network: &PyPolicyValueNetwork) -> PyTrajectory {
        let mut state = self.create_game_state(seed);
        PyTrajectory(self.inner.play_game_with_network(&mut state, seed, network))
    }

    /// Play multiple games in sequence using pure MCTS.
    fn play_games(&self, count: usize, base_seed: u64) -> Vec<PyTrajectory> {
        (0..count)
            .map(|i| {
                let seed = base_seed.wrapping_add(i as u64);
                self.play_game(seed)
            })
            .collect()
    }

    /// Play multiple games using MCTS guided by a neural network.
    fn play_games_with_network(
        &self,
        count: usize,
        base_seed: u64,
        network: &PyPolicyValueNetwork,
    ) -> Vec<PyTrajectory> {
        (0..count)
            .map(|i| {
                let seed = base_seed.wrapping_add(i as u64);
                self.play_game_with_network(seed, network)
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "SimpleGameWorker(players={}, life={}, hand={}, deck={})",
            self.player_count, self.starting_life, self.starting_hand_size, self.cards_per_player
        )
    }
}

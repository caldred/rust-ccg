//! Python bindings for the rust-ccg card game engine.
//!
//! This module provides PyO3 bindings for training neural networks with MCTS self-play.
//!
//! # Quick Start
//!
//! ```python
//! import rust_ccg as ccg
//!
//! # Create a self-play configuration
//! config = ccg.SelfPlayConfig(mcts_iterations=100, temperature=1.0)
//!
//! # Create a worker for SimpleGame
//! worker = ccg.SimpleGameWorker(player_count=2, config=config)
//!
//! # Play a game and collect trajectory
//! trajectory = worker.play_game(seed=42)
//!
//! # Convert to training samples
//! samples = trajectory.to_training_samples()
//! ```

use pyo3::prelude::*;

mod py_core;
mod py_games;
mod py_nn;
mod py_self_play;
mod py_training;

pub use py_core::*;
pub use py_games::*;
pub use py_nn::*;
pub use py_self_play::*;
pub use py_training::*;

/// rust-ccg: A card game engine for AlphaZero-style training.
///
/// This module provides:
/// - Game abstractions (SimpleGame for testing)
/// - MCTS self-play infrastructure
/// - Training data collection and buffering
/// - Neural network integration via Python callbacks
#[pymodule]
fn rust_ccg(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core types
    m.add_class::<PyPlayerId>()?;
    m.add_class::<PyAction>()?;
    m.add_class::<PyTemplateId>()?;

    // Neural network types
    m.add_class::<PyEncodedState>()?;
    m.add_class::<PyPolicyValueNetwork>()?;
    m.add_class::<PyUniformPolicy>()?;
    m.add_class::<PySimpleEncoder>()?;

    // Training types
    m.add_class::<PyStep>()?;
    m.add_class::<PyTrajectory>()?;
    m.add_class::<PyTrainingSample>()?;
    m.add_class::<PyExperienceBuffer>()?;
    m.add_class::<PyTrajectoryIterator>()?;

    // Self-play
    m.add_class::<PySelfPlayConfig>()?;
    m.add_class::<PySimpleGameWorker>()?;

    // Games
    m.add_class::<PySimpleGame>()?;

    Ok(())
}

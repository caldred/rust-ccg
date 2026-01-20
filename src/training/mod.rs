//! Training infrastructure for neural network integration.
//!
//! This module provides the data structures and self-play loop for
//! generating training data in an AlphaZero-style training pipeline.
//!
//! ## Overview
//!
//! - **Trajectory**: Records a complete game with states, policies, and outcome
//! - **ExperienceBuffer**: Collects and samples from trajectories
//! - **SelfPlayWorker**: Runs games using MCTS to generate trajectories
//!
//! ## Usage
//!
//! ```rust,ignore
//! use rust_ccg::training::{SelfPlayConfig, SelfPlayWorker, ExperienceBuffer};
//! use rust_ccg::nn::SimpleGameEncoder;
//!
//! // Set up self-play
//! let config = SelfPlayConfig::default()
//!     .with_mcts_iterations(800)
//!     .with_temperature(1.0);
//!
//! let encoder = Box::new(SimpleGameEncoder::new(2, 10));
//! let worker = SelfPlayWorker::new(engine, encoder, config);
//!
//! // Generate training data
//! let trajectory = worker.play_game(&mut state, seed);
//!
//! // Collect in buffer
//! let mut buffer = ExperienceBuffer::new(10000);
//! buffer.push(trajectory);
//!
//! // Sample training batch
//! let samples = buffer.sample_batch(32, rng_seed);
//! ```

pub mod self_play;
pub mod trajectory;

// Re-export main types
pub use self_play::{SelfPlayConfig, SelfPlayWorker};
pub use trajectory::{ExperienceBuffer, Step, Trajectory, TrainingSample};

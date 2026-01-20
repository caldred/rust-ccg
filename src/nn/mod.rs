//! Neural network integration for rust-ccg.
//!
//! This module provides traits and utilities for integrating neural networks
//! with MCTS for AlphaZero-style training.
//!
//! ## Overview
//!
//! - **Traits**: `PolicyNetwork`, `ValueNetwork`, `PolicyValueNetwork`
//! - **Encoding**: `StateEncoder` trait and `SimpleGameEncoder` implementation
//! - **Baseline**: `UniformPolicy`, `ZeroValue` for testing
//!
//! ## Usage
//!
//! ```rust,ignore
//! use rust_ccg::nn::{StateEncoder, SimpleGameEncoder, PolicyValueNetwork};
//!
//! // Create an encoder for your game
//! let encoder = SimpleGameEncoder::new(2, 10);
//!
//! // Encode game state for neural network input
//! let encoded = encoder.encode(&state, player);
//!
//! // Get predictions from network
//! let (policy, value) = network.predict(&encoded);
//! ```

pub mod encoder;
pub mod traits;

// Re-export main types
pub use encoder::{SimpleGameEncoder, StateEncoder, ZeroEncoder};
pub use traits::{
    EncodedState, PolicyNetwork, PolicyValueNetwork, UniformPolicy, UniformPolicyZeroValue,
    ValueNetwork, ZeroValue,
};

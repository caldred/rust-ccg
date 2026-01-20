//! Rules engine trait for game implementations.
//!
//! Games implement `RulesEngine` to define:
//! - Legal actions for each game state
//! - How actions modify state
//! - Win/loss conditions
//!
//! The core engine calls into `RulesEngine` but never interprets
//! game-specific concepts directly.

pub mod engine;

pub use engine::{GameResult, RulesEngine};

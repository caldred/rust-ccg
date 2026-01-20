//! Simple "War" game for testing the engine.
//!
//! A minimal game that validates the core engine:
//! - Each player starts with 20 life and a deck of cards
//! - Cards have a "power" value
//! - On your turn: draw a card OR play a card to damage an opponent
//! - First player to 0 life loses
//!
//! Supports 2-8 players to verify N-player generality.

mod game;

pub use game::{SimpleGame, SimpleGameBuilder};

//! Core engine types: entities, players, state, actions, RNG, configuration.
//!
//! This module contains the fundamental building blocks that are game-agnostic.
//! Games configure these via `GameConfig` rather than modifying the core.

pub mod entity;
pub mod player;
pub mod rng;
pub mod config;
pub mod action;
pub mod state;

pub use entity::EntityId;
pub use player::{PlayerId, PlayerMap};
pub use rng::{GameRng, GameRngState};
pub use config::{ZoneId, ZoneConfig, ZoneVisibility, TemplateId, TemplateConfig, PhaseId, GameConfig};
pub use action::{Action, ActionRecord};
pub use state::{PublicState, GameState};

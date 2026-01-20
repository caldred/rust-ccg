//! Zone system for card locations.
//!
//! Zones are **game-configured**, not hardcoded. Games define their zones
//! (hand, library, battlefield, market, etc.) via `ZoneConfig` at startup.
//!
//! ## Key Types
//!
//! - `ZoneId`: Opaque zone identifier (from `core::config`)
//! - `ZoneConfig`: Zone properties (visibility, ordering, ownership)
//! - `ZoneManager`: Card location tracking and movement
//! - `ZonePosition`: Position specifier for ordered zones

pub mod manager;

pub use manager::{ZoneManager, ZonePosition};

// Re-export zone types from core for convenience
pub use crate::core::config::{ZoneId, ZoneConfig, ZoneVisibility};

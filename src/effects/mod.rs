//! Effect system for card game actions.
//!
//! Effects are the building blocks of card abilities:
//! - `Effect`: Enumeration of game-agnostic effect types
//! - `TargetSpec`: How to select targets for effects
//! - `EffectResolver`: Executes effects on game state
//!
//! ## Design Philosophy
//!
//! The effect system is intentionally simple and game-agnostic.
//! Effects operate on generic concepts:
//! - Modify player state (life, resources)
//! - Move cards between zones
//! - Modify card state (counters, flags)
//!
//! Games define the meaning of these operations through their
//! state key conventions and zone configurations.

mod effect;
mod targeting;
mod resolver;

pub use effect::{Effect, EffectBatch};
pub use targeting::{TargetSpec, TargetFilter, TargetSelector};
pub use resolver::{EffectResolver, ResolverContext};

//! Stack and resolution system.
//!
//! This module provides flexible effect resolution supporting both:
//! - **Immediate resolution** (Hearthstone-style): Effects resolve as they happen
//! - **Priority stack** (MTG-style): Effects queue on a stack, LIFO resolution
//!
//! ## Design Philosophy
//!
//! Games choose their resolution mode by instantiating the appropriate
//! `ResolutionSystem` implementation. The engine provides the infrastructure;
//! games decide when and how effects resolve.
//!
//! ## Example Usage
//!
//! ```
//! use rust_ccg::core::{GameState, PlayerId};
//! use rust_ccg::effects::{Effect, EffectBatch, ResolverContext};
//! use rust_ccg::stack::{ImmediateResolution, ResolutionSystem, ResolutionStatus};
//!
//! // Create a Hearthstone-style resolver
//! let mut resolver = ImmediateResolution::new();
//!
//! // Queue some effects
//! let mut batch = EffectBatch::new();
//! batch.add_player(Effect::damage(3), PlayerId::new(1));
//!
//! // In immediate mode, effects resolve right away during process()
//! let mut state = GameState::new(2, 42);
//! let context = ResolverContext::simple(2);
//! let status = resolver.process(&mut state, &context);
//! assert!(matches!(status, ResolutionStatus::Complete));
//! ```

mod immediate;
mod priority;

pub use immediate::ImmediateResolution;
pub use priority::{PriorityStack, StackEntry, StackEntryId, StackSource};

use serde::{Deserialize, Serialize};

use crate::core::{Action, GameState, PlayerId};
use crate::effects::{EffectBatch, ResolverContext};
use crate::triggers::TriggeredEffect;

/// Status returned by resolution processing.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResolutionStatus {
    /// Resolution is complete, stack is empty.
    Complete,

    /// Waiting for a player to pass or respond.
    WaitingForPriority(PlayerId),

    /// Still processing (more effects to resolve).
    Processing,
}

/// Trait for resolution systems.
///
/// Games implement or use a resolution system to control how effects
/// are queued and resolved. The two main implementations are:
///
/// - [`ImmediateResolution`]: Effects resolve immediately (Hearthstone-style)
/// - [`PriorityStack`]: Effects queue on a stack with priority passing (MTG-style)
pub trait ResolutionSystem {
    /// Queue an action's effects for resolution.
    ///
    /// The `source_action` is stored for reference (e.g., "what caused this effect").
    fn queue_action(
        &mut self,
        source_action: Action,
        effects: EffectBatch,
        controller: PlayerId,
    );

    /// Queue a triggered effect for resolution.
    ///
    /// Triggered effects come from the trigger registry when events fire.
    fn queue_triggered(&mut self, triggered: TriggeredEffect);

    /// Process resolution until complete or waiting for input.
    ///
    /// Returns the current status:
    /// - `Complete`: All effects resolved, nothing left to do
    /// - `WaitingForPriority(player)`: Player must pass or respond
    /// - `Processing`: More effects to resolve (call again)
    fn process(&mut self, state: &mut GameState, context: &ResolverContext) -> ResolutionStatus;

    /// Check if resolution is complete (stack empty, no pending effects).
    fn is_complete(&self) -> bool;

    /// Get the player who currently has priority, if any.
    ///
    /// Returns `None` for immediate resolution mode or when no one has priority.
    fn priority_player(&self) -> Option<PlayerId>;

    /// Clear all pending effects and reset state.
    fn clear(&mut self);
}

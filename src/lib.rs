//! # rust-ccg v2
//!
//! A general-purpose card game engine optimized for RL/MCTS training.
//!
//! ## Design Principles
//!
//! 1. **Game-Agnostic**: No hardcoded zones, templates, phases, or card types.
//!    Games configure these at startup.
//!
//! 2. **N-Player First**: Every API takes `player_count` as context.
//!    No convenience methods that assume 2 players.
//!
//! 3. **Configuration Over Convention**: Games define their structure via
//!    `GameConfig`, `ZoneConfig`, `TemplateConfig`.
//!
//! ## Architecture
//!
//! - **Public-State MCTS**: Only expand nodes on our turns, sample opponent
//!   actions from learned policy. Action history provides consistency.
//!
//! - **Persistent Data Structures**: O(1) cloning via `im-rs` for MCTS.
//!
//! - **State Values**: All state uses `i64` for MCTS performance.
//!   See `State Value Encoding` in the plan for encoding other types.
//!
//! ## Modules
//!
//! - `core`: Entity IDs, players, state, actions, RNG, configuration
//! - `zones`: Zone system (game-configured, not hardcoded)
//! - `cards`: Card definitions and instances
//! - `rules`: RulesEngine trait for game implementations
//! - `effects`: Effect system for card abilities
//! - `triggers`: Event-driven trigger system
//! - `stack`: Resolution systems (immediate and priority-based)
//! - `mcts`: Monte Carlo Tree Search for AI

pub mod core;
pub mod zones;
pub mod cards;
pub mod rules;
pub mod effects;
pub mod triggers;
pub mod stack;
pub mod mcts;
pub mod games;

// Re-export commonly used types
pub use crate::core::{
    EntityId, PlayerId, PlayerMap,
    GameRng, GameRngState,
    ZoneId, ZoneConfig, ZoneVisibility,
    TemplateId, TemplateConfig, PhaseId, GameConfig,
    Action, ActionRecord,
    PublicState, GameState,
};

pub use crate::zones::{ZoneManager, ZonePosition};

pub use crate::cards::{
    CardId, CardTypeId, CardDefinition, CardInstance,
    CardRegistry, AttributeKey, AttributeValue, Attributes,
};

pub use crate::rules::{RulesEngine, GameResult};

pub use crate::effects::{Effect, EffectBatch, TargetSpec, TargetFilter, TargetSelector, EffectResolver, ResolverContext};

pub use crate::triggers::{
    EventTypeId, GameEvent, EventTypeConfig,
    TriggerCondition, ConditionContext, ConditionEvaluator,
    TriggerId, Trigger, TriggerRegistry, TriggerTiming, TriggeredEffect,
};

pub use crate::stack::{
    ResolutionStatus, ResolutionSystem,
    ImmediateResolution,
    PriorityStack, StackEntry, StackEntryId, StackSource,
};

pub use crate::mcts::{
    MCTSConfig, MCTSSearch, MCTSTree, MCTSNode, NodeId, Edge,
    SearchStats, TreeStats,
    SelectionPolicy, SimulationPolicy, OpponentPolicy,
    UCB1, PUCT, RandomSimulation, UniformOpponent,
};

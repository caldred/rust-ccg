//! Monte Carlo Tree Search for rust-ccg.
//!
//! ## Overview
//!
//! This module implements Public-State MCTS optimized for games with hidden
//! information. Key features:
//!
//! - **Public-State MCTS**: Nodes only expand on the searching player's turns
//! - **Opponent Modeling**: Opponent actions sampled from configurable policies
//! - **N-Player Support**: Works with any number of players
//! - **Configurable Policies**: Selection (UCB1/PUCT), simulation, opponent
//! - **Serializable**: Tree and config can be saved/loaded
//!
//! ## Usage
//!
//! ```rust
//! use rust_ccg::core::{GameState, PlayerId};
//! use rust_ccg::mcts::{MCTSConfig, MCTSSearch};
//! use rust_ccg::rules::RulesEngine;
//!
//! // Assuming you have a game engine implementing RulesEngine
//! fn example<E: RulesEngine + Clone>(engine: E, state: &mut GameState) {
//!     let config = MCTSConfig::default();
//!     let mut search = MCTSSearch::new(engine, config);
//!
//!     // Run 1000 iterations of MCTS
//!     // Note: Takes &mut GameState because cloning requires RNG forking
//!     if let Some(action) = search.search(state, PlayerId::new(0), 1000) {
//!         println!("Best action: {:?}", action);
//!     }
//!
//!     // Get action probabilities for training
//!     let probs = search.action_probabilities();
//!     for (action, prob) in probs {
//!         println!("{:?}: {:.2}%", action, prob * 100.0);
//!     }
//! }
//! ```
//!
//! ## Custom Policies
//!
//! You can customize the search behavior with different policies:
//!
//! ```rust,ignore
//! use rust_ccg::mcts::{MCTSSearch, MCTSConfig, PUCT};
//!
//! let search = MCTSSearch::new(engine, config)
//!     .with_selection(PUCT);  // Use PUCT instead of UCB1
//! ```

pub mod config;
pub mod node;
pub mod policy;
pub mod search;
pub mod stats;
pub mod tree;

// Re-export main types
pub use config::MCTSConfig;
pub use node::{Edge, MCTSNode, NodeId};
pub use policy::{
    OpponentPolicy, RandomSimulation, SelectionPolicy, SimulationPolicy,
    UCB1, PUCT, UniformOpponent,
};
pub use search::MCTSSearch;
pub use stats::SearchStats;
pub use tree::{MCTSTree, TreeStats};

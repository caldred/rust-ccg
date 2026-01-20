//! Core MCTS search algorithm.
//!
//! Implements Public-State MCTS where nodes are only expanded on the
//! searching player's turns. Opponent actions are sampled from a
//! configurable policy.

use std::time::Instant;

use crate::core::{Action, GameRng, GameState, PlayerId, PlayerMap};
use crate::rules::RulesEngine;

use super::config::MCTSConfig;
use super::node::{Edge, MCTSNode, NodeId};
use super::policy::{
    result_to_rewards, OpponentPolicy, RandomSimulation, SelectionPolicy, SimulationPolicy,
    UCB1, UniformOpponent,
};
use super::stats::SearchStats;
use super::tree::MCTSTree;

/// Main MCTS search context.
///
/// Generic over the rules engine type. Owns the search tree and
/// configuration, and provides methods to run searches.
pub struct MCTSSearch<E: RulesEngine> {
    /// The game rules engine.
    engine: E,

    /// Search configuration.
    config: MCTSConfig,

    /// The search tree.
    tree: MCTSTree,

    /// RNG for simulations.
    rng: GameRng,

    /// Selection policy.
    selection: Box<dyn SelectionPolicy>,

    /// Simulation policy.
    simulation: Box<dyn SimulationPolicy<E>>,

    /// Opponent modeling policy.
    opponent: Box<dyn OpponentPolicy<E>>,

    /// Search statistics.
    stats: SearchStats,
}

impl<E: RulesEngine + Clone> MCTSSearch<E> {
    /// Create a new MCTS search context.
    pub fn new(engine: E, config: MCTSConfig) -> Self {
        let player_count = engine.config().player_count;
        let rng = GameRng::new(config.seed);

        Self {
            engine,
            config: config.clone(),
            tree: MCTSTree::with_capacity(PlayerId::new(0), player_count, config.max_nodes),
            rng,
            selection: Box::new(UCB1),
            simulation: Box::new(RandomSimulation),
            opponent: Box::new(UniformOpponent),
            stats: SearchStats::default(),
        }
    }

    /// Set a custom selection policy.
    pub fn with_selection<S: SelectionPolicy + 'static>(mut self, selection: S) -> Self {
        self.selection = Box::new(selection);
        self
    }

    /// Set a custom simulation policy.
    pub fn with_simulation<S: SimulationPolicy<E> + 'static>(mut self, simulation: S) -> Self {
        self.simulation = Box::new(simulation);
        self
    }

    /// Set a custom opponent policy.
    pub fn with_opponent<O: OpponentPolicy<E> + 'static>(mut self, opponent: O) -> Self {
        self.opponent = Box::new(opponent);
        self
    }

    /// Run MCTS search for a given number of iterations.
    ///
    /// Returns the best action for the searching player.
    ///
    /// Note: Takes `&mut GameState` because cloning state requires forking
    /// the RNG for deterministic simulation branches.
    pub fn search(
        &mut self,
        state: &mut GameState,
        player: PlayerId,
        iterations: u32,
    ) -> Option<Action> {
        let start = Instant::now();
        self.stats.reset();

        // Initialize tree with root
        self.tree.reset(state.public.active_player);

        // Expand root node
        let root = self.tree.root();
        self.expand_node(root, state);

        // Check for terminal root
        if self.tree.get(root).is_terminal {
            return None;
        }

        // Check for single action (no choice)
        if self.tree.get(root).edges.len() == 1 {
            return Some(self.tree.get(root).edges[0].action.clone());
        }

        // Run iterations
        for _ in 0..iterations {
            let mut sim_state = state.clone_state();
            self.iteration(&mut sim_state, player);
            self.stats.iterations += 1;

            // Check node limit
            if self.tree.len() >= self.config.max_nodes {
                break;
            }
        }

        // Record time
        self.stats.time_us = start.elapsed().as_micros() as u64;

        // Select best action
        self.best_action(player)
    }

    /// Single MCTS iteration: select, expand, simulate, backpropagate.
    fn iteration(&mut self, state: &mut GameState, searching_player: PlayerId) {
        let mut path: Vec<(NodeId, usize)> = Vec::new();
        let mut current = self.tree.root();

        // === SELECTION ===
        loop {
            let node = self.tree.get(current);

            // Terminal node
            if node.is_terminal {
                if let Some(ref rewards) = node.terminal_reward {
                    self.backpropagate(&path, rewards.clone());
                }
                return;
            }

            // Depth limit
            if self.config.max_depth > 0 && node.depth >= self.config.max_depth as u16 {
                let rewards = super::policy::heuristic_eval(state, self.tree.player_count());
                self.backpropagate(&path, rewards);
                return;
            }

            // If not our turn, sample opponent action
            if node.to_move != searching_player {
                let opponent = node.to_move;

                if let Some(action) = self.sample_opponent_action(state, opponent) {
                    // Apply action
                    let mut engine = self.engine.clone();
                    engine.apply_action(state, opponent, &action);

                    // Find or create edge
                    let edge_idx = self.find_or_create_edge(current, &action);
                    path.push((current, edge_idx));

                    // Ensure child exists
                    let child = self.ensure_child(current, edge_idx, state);
                    current = child;
                    continue;
                } else {
                    // No legal moves - this is effectively terminal
                    let rewards = PlayerMap::with_value(self.tree.player_count(), 0.5);
                    self.backpropagate(&path, rewards);
                    return;
                }
            }

            // Our turn - use selection policy
            // Extract needed data before mutable operations
            let has_unexpanded = self.tree.get(current).has_unexpanded();
            let to_move = self.tree.get(current).to_move;

            // If unexpanded edges exist, expand one
            if has_unexpanded {
                let edge_idx = self.select_unexpanded(current);
                path.push((current, edge_idx));

                // Apply action
                let action = self.tree.get(current).edges[edge_idx].action.clone();
                let mut engine = self.engine.clone();
                engine.apply_action(state, to_move, &action);

                // Expand child
                let _child = self.expand_child(current, edge_idx, state);

                // Simulate from this state
                let rewards = self.simulate(state);
                self.stats.simulations += 1;
                self.backpropagate(&path, rewards);
                return;
            }

            // All edges expanded - select best
            let node = self.tree.get(current);
            let edge_idx = self.selection.select(node, searching_player, &self.config);
            path.push((current, edge_idx));

            // Apply action and descend
            let action = self.tree.get(current).edges[edge_idx].action.clone();
            let mut engine = self.engine.clone();
            engine.apply_action(state, to_move, &action);

            let child = self.tree.get(current).edges[edge_idx].child;
            if child.is_none() {
                // Should not happen if is_fully_expanded, but handle gracefully
                let _child = self.expand_child(current, edge_idx, state);
                let rewards = self.simulate(state);
                self.stats.simulations += 1;
                self.backpropagate(&path, rewards);
                return;
            }

            current = child;
        }
    }

    /// Expand a node with all legal actions.
    fn expand_node(&mut self, node_id: NodeId, state: &GameState) {
        let node = self.tree.get(node_id);
        let player = node.to_move;
        let player_count = self.tree.player_count();

        // Check for terminal
        if let Some(result) = self.engine.is_terminal(state) {
            let node = self.tree.get_mut(node_id);
            node.is_terminal = true;
            node.terminal_reward = Some(result_to_rewards(&result, player_count));
            return;
        }

        // Get legal actions
        let actions = self.engine.legal_actions(state, player);

        // Add edges
        let node = self.tree.get_mut(node_id);
        for action in actions {
            node.edges.push(Edge::new(action, player_count));
        }

        self.stats.nodes_expanded += 1;
    }

    /// Select an unexpanded edge randomly.
    fn select_unexpanded(&mut self, node_id: NodeId) -> usize {
        let node = self.tree.get(node_id);
        let unexpanded: Vec<usize> = node.unexpanded_edges().collect();

        if unexpanded.is_empty() {
            0
        } else if unexpanded.len() == 1 {
            unexpanded[0]
        } else {
            let idx = self.rng.gen_range_usize(0..unexpanded.len());
            unexpanded[idx]
        }
    }

    /// Expand a child node for the given edge.
    fn expand_child(&mut self, parent_id: NodeId, edge_idx: usize, state: &GameState) -> NodeId {
        let parent = self.tree.get(parent_id);
        let depth = parent.depth + 1;
        let to_move = state.public.active_player;

        // Track max depth
        if depth > self.stats.max_depth {
            self.stats.max_depth = depth;
        }

        let child = MCTSNode::new(parent_id, edge_idx as u16, to_move, depth);
        let child_id = self.tree.alloc(child);

        self.tree.get_mut(parent_id).edges[edge_idx].child = child_id;

        // Expand the new node
        self.expand_node(child_id, state);

        child_id
    }

    /// Ensure a child exists for the edge, creating if needed.
    fn ensure_child(&mut self, parent_id: NodeId, edge_idx: usize, state: &GameState) -> NodeId {
        let child = self.tree.get(parent_id).edges[edge_idx].child;
        if !child.is_none() {
            return child;
        }
        self.expand_child(parent_id, edge_idx, state)
    }

    /// Find or create an edge for an action.
    fn find_or_create_edge(&mut self, node_id: NodeId, action: &Action) -> usize {
        // First, search for existing edge
        let node = self.tree.get(node_id);
        for (i, edge) in node.edges.iter().enumerate() {
            if &edge.action == action {
                return i;
            }
        }

        // Edge doesn't exist - create it
        let player_count = self.tree.player_count();
        let node = self.tree.get_mut(node_id);
        node.edges.push(Edge::new(action.clone(), player_count));
        node.edges.len() - 1
    }

    /// Sample an opponent action.
    fn sample_opponent_action(&mut self, state: &GameState, opponent: PlayerId) -> Option<Action> {
        self.opponent.choose_action(&self.engine, state, opponent, &mut self.rng)
    }

    /// Run a simulation from the current state.
    fn simulate(&mut self, state: &mut GameState) -> PlayerMap<f64> {
        let mut sim_rng = self.rng.fork();
        let mut engine = self.engine.clone();
        self.simulation.simulate(
            &mut engine,
            state,
            &mut sim_rng,
            self.config.max_depth,
        )
    }

    /// Backpropagate rewards through the path.
    fn backpropagate(&mut self, path: &[(NodeId, usize)], rewards: PlayerMap<f64>) {
        let player_count = self.tree.player_count();

        for &(node_id, edge_idx) in path.iter().rev() {
            let node = self.tree.get_mut(node_id);
            node.visits += 1;

            let edge = &mut node.edges[edge_idx];
            edge.visits += 1;

            for player in PlayerId::all(player_count) {
                edge.total_reward[player] += rewards[player];
            }
        }

        // Update root visits
        self.tree.root_node_mut().visits += 1;
    }

    /// Select the best action from the root.
    fn best_action(&self, _player: PlayerId) -> Option<Action> {
        let root = self.tree.root_node();

        if root.edges.is_empty() {
            return None;
        }

        if self.config.temperature <= 0.0 {
            // Greedy: select most visited
            root.best_edge_by_visits().map(|e| e.action.clone())
        } else {
            // Temperature-based sampling
            let visits: Vec<f32> = root.edges.iter().map(|e| e.visits as f32).collect();
            let weights: Vec<f32> = visits
                .iter()
                .map(|&v| (v / self.config.temperature as f32).exp())
                .collect();

            let mut rng = self.rng.clone();
            rng.choose_weighted(&weights)
                .map(|idx| root.edges[idx].action.clone())
        }
    }

    /// Get search statistics.
    #[must_use]
    pub fn stats(&self) -> &SearchStats {
        &self.stats
    }

    /// Get the search tree.
    #[must_use]
    pub fn tree(&self) -> &MCTSTree {
        &self.tree
    }

    /// Get action visit counts from root (for training).
    ///
    /// Returns (action, visit_count) pairs.
    pub fn action_visits(&self) -> Vec<(Action, u32)> {
        self.tree
            .root_node()
            .edges
            .iter()
            .map(|e| (e.action.clone(), e.visits))
            .collect()
    }

    /// Get action probabilities from root (for training).
    ///
    /// Returns (action, probability) pairs where probabilities sum to ~1.0.
    pub fn action_probabilities(&self) -> Vec<(Action, f64)> {
        let root = self.tree.root_node();
        let total: u32 = root.edges.iter().map(|e| e.visits).sum();

        if total == 0 {
            let uniform = 1.0 / root.edges.len().max(1) as f64;
            return root
                .edges
                .iter()
                .map(|e| (e.action.clone(), uniform))
                .collect();
        }

        root.edges
            .iter()
            .map(|e| (e.action.clone(), e.visits as f64 / total as f64))
            .collect()
    }

    /// Get the engine reference.
    pub fn engine(&self) -> &E {
        &self.engine
    }

    /// Get the configuration.
    pub fn config(&self) -> &MCTSConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TemplateId;

    // Minimal test engine for unit tests
    struct TestEngine {
        config: crate::core::GameConfig,
        terminal_at: Option<u32>,
        action_count: usize,
    }

    impl TestEngine {
        fn new(player_count: usize) -> Self {
            Self {
                config: crate::core::GameConfig::new(player_count),
                terminal_at: None,
                action_count: 3,
            }
        }

        fn terminal_after(mut self, turns: u32) -> Self {
            self.terminal_at = Some(turns);
            self
        }
    }

    impl Clone for TestEngine {
        fn clone(&self) -> Self {
            Self {
                config: self.config.clone(),
                terminal_at: self.terminal_at,
                action_count: self.action_count,
            }
        }
    }

    impl RulesEngine for TestEngine {
        fn config(&self) -> &crate::core::GameConfig {
            &self.config
        }

        fn legal_templates(&self, _state: &GameState, _player: PlayerId) -> Vec<TemplateId> {
            (0..self.action_count as u16).map(TemplateId::new).collect()
        }

        fn legal_pointers(
            &self,
            _state: &GameState,
            _player: PlayerId,
            _template: TemplateId,
            _prior: &[crate::core::EntityId],
        ) -> Vec<crate::core::EntityId> {
            vec![] // No pointers needed
        }

        fn apply_action(&mut self, state: &mut GameState, _player: PlayerId, _action: &Action) {
            state.public.turn_number += 1;
            // Alternate players
            let next = (state.public.active_player.0 + 1) % self.config.player_count as u8;
            state.public.active_player = PlayerId::new(next);
        }

        fn is_terminal(&self, state: &GameState) -> Option<crate::rules::GameResult> {
            if let Some(limit) = self.terminal_at {
                if state.public.turn_number >= limit {
                    return Some(crate::rules::GameResult::Winner(PlayerId::new(0)));
                }
            }
            None
        }
    }

    #[test]
    fn test_search_returns_action() {
        let engine = TestEngine::new(2).terminal_after(10);
        let mut state = GameState::new(2, 42);
        let config = MCTSConfig::default();

        let mut search = MCTSSearch::new(engine, config);
        let action = search.search(&mut state, PlayerId::new(0), 100);

        assert!(action.is_some());
    }

    #[test]
    fn test_search_stats() {
        let engine = TestEngine::new(2).terminal_after(10);
        let mut state = GameState::new(2, 42);
        let config = MCTSConfig::default();

        let mut search = MCTSSearch::new(engine, config);
        search.search(&mut state, PlayerId::new(0), 50);

        let stats = search.stats();
        assert_eq!(stats.iterations, 50);
        assert!(stats.simulations > 0);
        assert!(stats.nodes_expanded > 0);
    }

    #[test]
    fn test_search_deterministic() {
        let engine1 = TestEngine::new(2).terminal_after(10);
        let engine2 = TestEngine::new(2).terminal_after(10);
        let mut state1 = GameState::new(2, 42);
        let mut state2 = GameState::new(2, 42);

        let config = MCTSConfig::default().with_seed(12345);

        let mut search1 = MCTSSearch::new(engine1, config.clone());
        let mut search2 = MCTSSearch::new(engine2, config);

        let action1 = search1.search(&mut state1, PlayerId::new(0), 100);
        let action2 = search2.search(&mut state2, PlayerId::new(0), 100);

        assert_eq!(action1, action2);
    }

    #[test]
    fn test_action_probabilities() {
        let engine = TestEngine::new(2).terminal_after(10);
        let mut state = GameState::new(2, 42);
        let config = MCTSConfig::default();

        let mut search = MCTSSearch::new(engine, config);
        search.search(&mut state, PlayerId::new(0), 100);

        let probs = search.action_probabilities();

        // Should have probabilities that sum to ~1.0
        let sum: f64 = probs.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_tree_growth() {
        let engine = TestEngine::new(2).terminal_after(20);
        let mut state = GameState::new(2, 42);
        let config = MCTSConfig::default();

        let mut search = MCTSSearch::new(engine, config);
        search.search(&mut state, PlayerId::new(0), 200);

        let tree_stats = search.tree().stats();
        assert!(tree_stats.node_count > 1);
        assert!(tree_stats.max_depth > 0);
    }
}

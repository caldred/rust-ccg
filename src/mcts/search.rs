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

    /// Set prior probabilities on root edges from a neural network policy.
    ///
    /// This allows neural network guidance for MCTS exploration.
    /// Priors should be set after calling `search()` has initialized the tree,
    /// or the root can be manually expanded first.
    ///
    /// # Arguments
    /// - `priors`: Slice of (action, prior_probability) pairs
    ///
    /// # Returns
    /// Number of edges that were updated
    pub fn set_root_priors(&mut self, priors: &[(Action, f32)]) -> usize {
        let root = self.tree.root();
        let node = self.tree.get_mut(root);
        let mut updated = 0;

        for edge in node.edges.iter_mut() {
            if let Some((_, prior)) = priors.iter().find(|(a, _)| a == &edge.action) {
                edge.prior = *prior;
                updated += 1;
            }
        }

        updated
    }

    /// Get the current root priors.
    ///
    /// Returns (action, prior) pairs for all root edges.
    pub fn root_priors(&self) -> Vec<(Action, f32)> {
        self.tree
            .root_node()
            .edges
            .iter()
            .map(|e| (e.action.clone(), e.prior))
            .collect()
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

    #[test]
    fn test_set_root_priors() {
        let engine = TestEngine::new(2).terminal_after(10);
        let mut state = GameState::new(2, 42);
        let config = MCTSConfig::default();

        let mut search = MCTSSearch::new(engine, config);
        search.search(&mut state, PlayerId::new(0), 10);

        // Get actions from root
        let actions: Vec<_> = search.action_visits().into_iter().map(|(a, _)| a).collect();

        // Set custom priors
        let priors: Vec<_> = actions
            .iter()
            .enumerate()
            .map(|(i, a)| (a.clone(), (i + 1) as f32 * 0.1))
            .collect();

        let updated = search.set_root_priors(&priors);
        assert_eq!(updated, actions.len());

        // Verify priors were set
        let root_priors = search.root_priors();
        for (action, prior) in &root_priors {
            let expected = priors.iter().find(|(a, _)| a == action).map(|(_, p)| *p);
            assert_eq!(Some(*prior), expected);
        }
    }

    #[test]
    fn test_set_root_priors_partial_match() {
        let engine = TestEngine::new(2).terminal_after(10);
        let mut state = GameState::new(2, 42);
        let config = MCTSConfig::default();

        let mut search = MCTSSearch::new(engine, config);
        search.search(&mut state, PlayerId::new(0), 10);

        // Only set prior for one action
        let action = Action::new(TemplateId::new(0));
        let priors = vec![(action, 0.9)];

        let updated = search.set_root_priors(&priors);
        assert_eq!(updated, 1);
    }

    #[test]
    fn test_set_root_priors_no_match() {
        let engine = TestEngine::new(2).terminal_after(10);
        let mut state = GameState::new(2, 42);
        let config = MCTSConfig::default();

        let mut search = MCTSSearch::new(engine, config);
        search.search(&mut state, PlayerId::new(0), 10);

        // Set prior for non-existent action
        let action = Action::new(TemplateId::new(999));
        let priors = vec![(action, 0.9)];

        let updated = search.set_root_priors(&priors);
        assert_eq!(updated, 0);
    }

    #[test]
    fn test_root_priors_default() {
        let engine = TestEngine::new(2).terminal_after(10);
        let mut state = GameState::new(2, 42);
        let config = MCTSConfig::default();

        let mut search = MCTSSearch::new(engine, config);
        search.search(&mut state, PlayerId::new(0), 10);

        // Default priors should all be 1.0
        let priors = search.root_priors();
        assert!(!priors.is_empty());
        for (_, prior) in &priors {
            assert_eq!(*prior, 1.0);
        }
    }

    #[test]
    fn test_root_priors_empty_tree() {
        let engine = TestEngine::new(2).terminal_after(0); // Terminal immediately
        let mut state = GameState::new(2, 42);
        let config = MCTSConfig::default();

        let mut search = MCTSSearch::new(engine, config);
        search.search(&mut state, PlayerId::new(0), 10);

        // Should have no priors (terminal state)
        let priors = search.root_priors();
        assert!(priors.is_empty());
    }

    #[test]
    fn test_action_visits() {
        let engine = TestEngine::new(2).terminal_after(10);
        let mut state = GameState::new(2, 42);
        let config = MCTSConfig::default();

        let mut search = MCTSSearch::new(engine, config);
        search.search(&mut state, PlayerId::new(0), 100);

        let visits = search.action_visits();

        // Should have visits for each action
        assert!(!visits.is_empty());

        // Total visits should be close to iterations
        let total: u32 = visits.iter().map(|(_, v)| v).sum();
        assert!(total > 0);
    }

    #[test]
    fn test_engine_accessor() {
        let engine = TestEngine::new(2).terminal_after(10);
        let config = MCTSConfig::default();

        let search = MCTSSearch::new(engine, config);

        // Verify engine is accessible
        assert_eq!(search.engine().config().player_count, 2);
    }

    #[test]
    fn test_config_accessor() {
        let engine = TestEngine::new(2).terminal_after(10);
        let config = MCTSConfig::default()
            .with_exploration(3.0)
            .with_seed(999);

        let search = MCTSSearch::new(engine, config);

        assert_eq!(search.config().exploration_constant, 3.0);
        assert_eq!(search.config().seed, 999);
    }

    #[test]
    fn test_action_probabilities_no_visits() {
        let engine = TestEngine::new(2).terminal_after(10);
        let mut state = GameState::new(2, 42);
        let config = MCTSConfig::default();

        let mut search = MCTSSearch::new(engine, config);
        // Only expand root, don't run full iterations
        search.search(&mut state, PlayerId::new(0), 1);

        let probs = search.action_probabilities();

        // Should still have probabilities (uniform if no visits)
        assert!(!probs.is_empty());
        let sum: f64 = probs.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_search_with_custom_policies() {
        use crate::mcts::PUCT;

        let engine = TestEngine::new(2).terminal_after(10);
        let mut state = GameState::new(2, 42);
        let config = MCTSConfig::default();

        let mut search = MCTSSearch::new(engine, config).with_selection(PUCT);
        let action = search.search(&mut state, PlayerId::new(0), 50);

        assert!(action.is_some());
    }

    #[test]
    fn test_search_four_player() {
        let engine = TestEngine::new(4).terminal_after(20);
        let mut state = GameState::new(4, 42);
        let config = MCTSConfig::default();

        let mut search = MCTSSearch::new(engine, config);
        let action = search.search(&mut state, PlayerId::new(0), 100);

        assert!(action.is_some());
    }

    // =========================================================================
    // Sign/Perspective correctness tests
    // =========================================================================

    /// Test engine where action 0 = player 0 wins, action 1 = player 1 wins
    struct AdversarialEngine {
        config: crate::core::GameConfig,
    }

    impl AdversarialEngine {
        fn new() -> Self {
            Self {
                config: crate::core::GameConfig::new(2),
            }
        }
    }

    impl Clone for AdversarialEngine {
        fn clone(&self) -> Self {
            Self {
                config: self.config.clone(),
            }
        }
    }

    impl RulesEngine for AdversarialEngine {
        fn config(&self) -> &crate::core::GameConfig {
            &self.config
        }

        fn legal_templates(&self, state: &GameState, _player: PlayerId) -> Vec<TemplateId> {
            // Only root has choices; after one action it's terminal
            // turn_number starts at 1
            if state.public.turn_number == 1 {
                vec![TemplateId::new(0), TemplateId::new(1)]
            } else {
                vec![]
            }
        }

        fn legal_pointers(
            &self,
            _state: &GameState,
            _player: PlayerId,
            _template: TemplateId,
            _prior: &[crate::core::EntityId],
        ) -> Vec<crate::core::EntityId> {
            vec![]
        }

        fn apply_action(&mut self, state: &mut GameState, _player: PlayerId, action: &Action) {
            // Store which action was taken in player state
            state.public.set_player_state(PlayerId::new(0), "chosen", action.template.0 as i64);
            state.public.turn_number += 1;
        }

        fn is_terminal(&self, state: &GameState) -> Option<crate::rules::GameResult> {
            // Turn starts at 1, increments after action
            if state.public.turn_number > 1 {
                let chosen = state.public.get_player_state(PlayerId::new(0), "chosen", 0);
                if chosen == 0 {
                    // Action 0 = Player 0 wins
                    Some(crate::rules::GameResult::Winner(PlayerId::new(0)))
                } else {
                    // Action 1 = Player 1 wins
                    Some(crate::rules::GameResult::Winner(PlayerId::new(1)))
                }
            } else {
                None
            }
        }
    }

    #[test]
    fn test_mcts_finds_winning_move_for_player0() {
        // Player 0 should choose action 0 (which makes them win)
        let engine = AdversarialEngine::new();
        let mut state = GameState::new(2, 42);
        let config = MCTSConfig::default();

        let mut search = MCTSSearch::new(engine, config);
        let action = search.search(&mut state, PlayerId::new(0), 100);

        // MCTS should find that action 0 wins for player 0
        assert_eq!(action, Some(Action::new(TemplateId::new(0))));

        // Verify Q-values: action 0 should have higher value for player 0
        let root = search.tree().root_node();
        let edge0 = root.edges.iter().find(|e| e.action.template == TemplateId::new(0)).unwrap();
        let edge1 = root.edges.iter().find(|e| e.action.template == TemplateId::new(1)).unwrap();

        // Action 0 should have Q=1.0 for player 0 (always wins)
        // Action 1 should have Q=0.0 for player 0 (always loses)
        assert!(
            edge0.mean_reward(PlayerId::new(0)) > edge1.mean_reward(PlayerId::new(0)),
            "Action 0 Q-value ({}) should be > Action 1 Q-value ({}) for Player 0",
            edge0.mean_reward(PlayerId::new(0)),
            edge1.mean_reward(PlayerId::new(0))
        );

        // Visit counts should strongly favor action 0
        assert!(
            edge0.visits > edge1.visits,
            "Action 0 should have more visits ({}) than Action 1 ({})",
            edge0.visits,
            edge1.visits
        );
    }

    #[test]
    fn test_mcts_q_values_correct_per_player() {
        let engine = AdversarialEngine::new();
        let mut state = GameState::new(2, 42);
        let config = MCTSConfig::default();

        let mut search = MCTSSearch::new(engine, config);
        search.search(&mut state, PlayerId::new(0), 100);

        let root = search.tree().root_node();
        let edge0 = root.edges.iter().find(|e| e.action.template == TemplateId::new(0)).unwrap();
        let edge1 = root.edges.iter().find(|e| e.action.template == TemplateId::new(1)).unwrap();

        // For Action 0: Player 0 wins, so Q(p0)=1.0, Q(p1)=0.0
        assert!(
            (edge0.mean_reward(PlayerId::new(0)) - 1.0).abs() < 0.01,
            "Action 0: Player 0 Q-value should be ~1.0, got {}",
            edge0.mean_reward(PlayerId::new(0))
        );
        assert!(
            edge0.mean_reward(PlayerId::new(1)).abs() < 0.01,
            "Action 0: Player 1 Q-value should be ~0.0, got {}",
            edge0.mean_reward(PlayerId::new(1))
        );

        // For Action 1: Player 1 wins, so Q(p0)=0.0, Q(p1)=1.0
        assert!(
            edge1.mean_reward(PlayerId::new(0)).abs() < 0.01,
            "Action 1: Player 0 Q-value should be ~0.0, got {}",
            edge1.mean_reward(PlayerId::new(0))
        );
        assert!(
            (edge1.mean_reward(PlayerId::new(1)) - 1.0).abs() < 0.01,
            "Action 1: Player 1 Q-value should be ~1.0, got {}",
            edge1.mean_reward(PlayerId::new(1))
        );
    }

    /// Two-move game to test alternating perspectives
    /// Turn 0: Player 0 picks A or B
    /// Turn 1: Player 1 picks X or Y
    /// Outcome depends on combination
    struct TwoMoveGame {
        config: crate::core::GameConfig,
    }

    impl TwoMoveGame {
        fn new() -> Self {
            Self {
                config: crate::core::GameConfig::new(2),
            }
        }
    }

    impl Clone for TwoMoveGame {
        fn clone(&self) -> Self {
            Self {
                config: self.config.clone(),
            }
        }
    }

    impl RulesEngine for TwoMoveGame {
        fn config(&self) -> &crate::core::GameConfig {
            &self.config
        }

        fn legal_templates(&self, state: &GameState, _player: PlayerId) -> Vec<TemplateId> {
            // Turn starts at 1, so turns 1 and 2 have actions
            if state.public.turn_number <= 2 {
                vec![TemplateId::new(0), TemplateId::new(1)]
            } else {
                vec![]
            }
        }

        fn legal_pointers(
            &self,
            _state: &GameState,
            _player: PlayerId,
            _template: TemplateId,
            _prior: &[crate::core::EntityId],
        ) -> Vec<crate::core::EntityId> {
            vec![]
        }

        fn apply_action(&mut self, state: &mut GameState, player: PlayerId, action: &Action) {
            // Store choices
            let key = if player == PlayerId::new(0) { "p0_choice" } else { "p1_choice" };
            state.public.set_player_state(PlayerId::new(0), key, action.template.0 as i64);
            state.public.turn_number += 1;
            // Alternate players
            state.public.active_player = PlayerId::new((player.0 + 1) % 2);
        }

        fn is_terminal(&self, state: &GameState) -> Option<crate::rules::GameResult> {
            // Turn starts at 1, after 2 actions turn is 3
            if state.public.turn_number > 2 {
                let p0_choice = state.public.get_player_state(PlayerId::new(0), "p0_choice", 0);
                let p1_choice = state.public.get_player_state(PlayerId::new(0), "p1_choice", 0);

                // Payoff matrix (from player 0's perspective):
                // P0\P1  |  0   |  1
                // -------|------|------
                //   0    | Win  | Draw
                //   1    | Draw | Lose
                //
                // P0 should choose 0 (dominates): wins if P1=0, draws if P1=1
                // P0 choosing 1 is worse: draws if P1=0, loses if P1=1

                let result = match (p0_choice, p1_choice) {
                    (0, 0) => crate::rules::GameResult::Winner(PlayerId::new(0)),
                    (0, 1) => crate::rules::GameResult::Draw,
                    (1, 0) => crate::rules::GameResult::Draw,
                    (1, 1) => crate::rules::GameResult::Winner(PlayerId::new(1)),
                    _ => crate::rules::GameResult::Draw,
                };
                Some(result)
            } else {
                None
            }
        }
    }

    #[test]
    fn test_mcts_two_move_game_finds_dominant_strategy() {
        // Player 0 should choose action 0 (dominates action 1)
        let engine = TwoMoveGame::new();
        let mut state = GameState::new(2, 42);
        let config = MCTSConfig::default();

        let mut search = MCTSSearch::new(engine, config);
        let action = search.search(&mut state, PlayerId::new(0), 200);

        // With uniform opponent policy (50% each):
        // Action 0 EV = 0.5 * 1.0 + 0.5 * 0.5 = 0.75
        // Action 1 EV = 0.5 * 0.5 + 0.5 * 0.0 = 0.25
        // So MCTS should prefer action 0
        assert_eq!(
            action,
            Some(Action::new(TemplateId::new(0))),
            "Player 0 should choose dominant strategy (action 0)"
        );
    }

    #[test]
    fn test_mcts_backpropagates_correctly_through_opponent_moves() {
        let engine = TwoMoveGame::new();
        let mut state = GameState::new(2, 42);
        let config = MCTSConfig::default();

        let mut search = MCTSSearch::new(engine, config);
        search.search(&mut state, PlayerId::new(0), 500);

        let root = search.tree().root_node();
        let edge0 = root.edges.iter().find(|e| e.action.template == TemplateId::new(0)).unwrap();
        let edge1 = root.edges.iter().find(|e| e.action.template == TemplateId::new(1)).unwrap();

        // Expected values with uniform opponent:
        // Action 0: EV(p0) = 0.5*1 + 0.5*0.5 = 0.75, EV(p1) = 0.5*0 + 0.5*0.5 = 0.25
        // Action 1: EV(p0) = 0.5*0.5 + 0.5*0 = 0.25, EV(p1) = 0.5*0.5 + 0.5*1 = 0.75

        let q0_a0 = edge0.mean_reward(PlayerId::new(0));
        let q0_a1 = edge1.mean_reward(PlayerId::new(0));

        assert!(
            q0_a0 > q0_a1,
            "Action 0 should have higher Q for player 0: {} vs {}",
            q0_a0,
            q0_a1
        );

        // Check approximate values (allow some variance from sampling)
        assert!(
            (q0_a0 - 0.75).abs() < 0.15,
            "Action 0 Q(p0) should be ~0.75, got {}",
            q0_a0
        );
        assert!(
            (q0_a1 - 0.25).abs() < 0.15,
            "Action 1 Q(p0) should be ~0.25, got {}",
            q0_a1
        );
    }
}

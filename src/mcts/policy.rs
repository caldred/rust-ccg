//! MCTS policies for selection, simulation, and opponent modeling.
//!
//! Policies are trait-based to allow customization:
//! - `SelectionPolicy`: How to choose which child to explore (UCB1, PUCT)
//! - `SimulationPolicy`: How to run rollouts (random, heuristic, neural)
//! - `OpponentPolicy`: How to model opponent behavior

use crate::core::{Action, GameRng, GameState, PlayerId, PlayerMap};
use crate::rules::{GameResult, RulesEngine};

use super::config::MCTSConfig;
use super::node::MCTSNode;

// =============================================================================
// Selection Policy
// =============================================================================

/// Policy for selecting which child node to explore.
pub trait SelectionPolicy: Send + Sync {
    /// Select the best edge index from a node.
    ///
    /// Returns the index of the edge to follow.
    fn select(&self, node: &MCTSNode, player: PlayerId, config: &MCTSConfig) -> usize;
}

/// UCB1 (Upper Confidence Bound) selection policy.
///
/// Balances exploitation (high reward) with exploration (low visits).
/// Formula: Q(a) + c * sqrt(ln(N) / n(a))
#[derive(Clone, Debug, Default)]
pub struct UCB1;

impl SelectionPolicy for UCB1 {
    fn select(&self, node: &MCTSNode, player: PlayerId, config: &MCTSConfig) -> usize {
        if node.edges.is_empty() {
            return 0;
        }

        let ln_parent = (node.visits.max(1) as f64).ln();

        node.edges
            .iter()
            .enumerate()
            .map(|(i, edge)| {
                let exploitation = edge.mean_reward(player);
                let exploration = if edge.visits == 0 {
                    f64::INFINITY
                } else {
                    config.exploration_constant * (ln_parent / edge.visits as f64).sqrt()
                };
                (i, exploitation + exploration)
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

/// PUCT selection policy (Predictor + UCB for Trees).
///
/// Uses prior probabilities from a policy network.
/// Formula: Q(a) + c * P(a) * sqrt(N) / (1 + n(a))
#[derive(Clone, Debug, Default)]
pub struct PUCT;

impl SelectionPolicy for PUCT {
    fn select(&self, node: &MCTSNode, player: PlayerId, config: &MCTSConfig) -> usize {
        if node.edges.is_empty() {
            return 0;
        }

        let sqrt_parent = (node.visits.max(1) as f64).sqrt();

        node.edges
            .iter()
            .enumerate()
            .map(|(i, edge)| {
                let q = edge.mean_reward(player);
                let u = config.exploration_constant
                    * edge.prior as f64
                    * sqrt_parent
                    / (1.0 + edge.visits as f64);
                (i, q + u)
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

// =============================================================================
// Simulation Policy
// =============================================================================

/// Policy for running simulations (rollouts) from a leaf node.
pub trait SimulationPolicy<E: RulesEngine>: Send + Sync {
    /// Run a simulation from the given state, returning rewards per player.
    ///
    /// The state is modified during simulation.
    fn simulate(
        &self,
        engine: &mut E,
        state: &mut GameState,
        rng: &mut GameRng,
        max_depth: u32,
    ) -> PlayerMap<f64>;
}

/// Random simulation policy.
///
/// Plays random legal actions until terminal or depth limit.
#[derive(Clone, Debug, Default)]
pub struct RandomSimulation;

impl<E: RulesEngine> SimulationPolicy<E> for RandomSimulation {
    fn simulate(
        &self,
        engine: &mut E,
        state: &mut GameState,
        rng: &mut GameRng,
        max_depth: u32,
    ) -> PlayerMap<f64> {
        let player_count = state.player_count();
        let mut depth = 0;

        loop {
            // Check for terminal
            if let Some(result) = engine.is_terminal(state) {
                return result_to_rewards(&result, player_count);
            }

            // Check depth limit
            if max_depth > 0 && depth >= max_depth {
                return heuristic_eval(state, player_count);
            }

            let active = state.public.active_player;
            let actions = engine.legal_actions(state, active);

            if actions.is_empty() {
                // No legal actions - draw
                return PlayerMap::with_value(player_count, 0.5);
            }

            // Random action
            let idx = rng.gen_range_usize(0..actions.len());
            engine.apply_action(state, active, &actions[idx]);

            depth += 1;
        }
    }
}

// =============================================================================
// Opponent Policy
// =============================================================================

/// Policy for sampling opponent actions during tree traversal.
pub trait OpponentPolicy<E: RulesEngine>: Send + Sync {
    /// Choose an action for an opponent.
    ///
    /// Returns `None` if no legal actions exist.
    fn choose_action(
        &self,
        engine: &E,
        state: &GameState,
        opponent: PlayerId,
        rng: &mut GameRng,
    ) -> Option<Action>;
}

/// Uniform random opponent policy.
///
/// Selects uniformly from legal actions.
#[derive(Clone, Debug, Default)]
pub struct UniformOpponent;

impl<E: RulesEngine> OpponentPolicy<E> for UniformOpponent {
    fn choose_action(
        &self,
        engine: &E,
        state: &GameState,
        opponent: PlayerId,
        rng: &mut GameRng,
    ) -> Option<Action> {
        let actions = engine.legal_actions(state, opponent);
        if actions.is_empty() {
            return None;
        }
        let idx = rng.gen_range_usize(0..actions.len());
        Some(actions[idx].clone())
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Convert a game result to per-player rewards.
pub fn result_to_rewards(result: &GameResult, player_count: usize) -> PlayerMap<f64> {
    PlayerMap::new(player_count, |player| match result {
        GameResult::Winner(winner) => {
            if *winner == player {
                1.0
            } else {
                0.0
            }
        }
        GameResult::Winners(winners) => {
            if winners.contains(&player) {
                1.0
            } else {
                0.0
            }
        }
        GameResult::Draw => 0.5,
    })
}

/// Simple heuristic evaluation based on life totals.
///
/// Returns relative life proportion as reward estimate.
pub fn heuristic_eval(state: &GameState, player_count: usize) -> PlayerMap<f64> {
    let lives: Vec<i64> = PlayerId::all(player_count)
        .map(|p| state.public.get_player_state(p, "life", 0).max(0))
        .collect();

    let total: i64 = lives.iter().sum();
    if total <= 0 {
        return PlayerMap::with_value(player_count, 0.5);
    }

    PlayerMap::new(player_count, |player| {
        lives[player.index()] as f64 / total as f64
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TemplateId;
    use crate::mcts::node::Edge;

    fn make_test_node() -> MCTSNode {
        let mut node = MCTSNode::root(PlayerId::new(0));

        // Edge 0: high reward, many visits
        let mut e0 = Edge::new(Action::new(TemplateId::new(1)), 2);
        e0.visits = 100;
        e0.total_reward[PlayerId::new(0)] = 80.0;

        // Edge 1: lower reward, fewer visits (should explore)
        let mut e1 = Edge::new(Action::new(TemplateId::new(2)), 2);
        e1.visits = 10;
        e1.total_reward[PlayerId::new(0)] = 7.0;

        // Edge 2: unvisited (infinite exploration bonus)
        let e2 = Edge::new(Action::new(TemplateId::new(3)), 2);

        node.edges.push(e0);
        node.edges.push(e1);
        node.edges.push(e2);
        node.visits = 111;

        node
    }

    #[test]
    fn test_ucb1_selects_unvisited() {
        let node = make_test_node();
        let config = MCTSConfig::default();
        let ucb1 = UCB1;

        // Should select unvisited edge (index 2) due to infinite exploration bonus
        let selected = ucb1.select(&node, PlayerId::new(0), &config);
        assert_eq!(selected, 2);
    }

    #[test]
    fn test_ucb1_all_visited() {
        let mut node = make_test_node();
        node.edges[2].visits = 5;
        node.edges[2].total_reward[PlayerId::new(0)] = 2.0;

        let config = MCTSConfig::default();
        let ucb1 = UCB1;

        // All visited - should balance exploitation and exploration
        let selected = ucb1.select(&node, PlayerId::new(0), &config);
        // The exact choice depends on the formula, but it should be valid
        assert!(selected < 3);
    }

    #[test]
    fn test_puct_uses_prior() {
        let mut node = MCTSNode::root(PlayerId::new(0));

        // Equal visits and rewards, but different priors
        let mut e0 = Edge::with_prior(Action::new(TemplateId::new(1)), 2, 0.1);
        e0.visits = 10;
        e0.total_reward[PlayerId::new(0)] = 5.0;

        let mut e1 = Edge::with_prior(Action::new(TemplateId::new(2)), 2, 0.9);
        e1.visits = 10;
        e1.total_reward[PlayerId::new(0)] = 5.0;

        node.edges.push(e0);
        node.edges.push(e1);
        node.visits = 20;

        let config = MCTSConfig::default();
        let puct = PUCT;

        // Should prefer edge with higher prior
        let selected = puct.select(&node, PlayerId::new(0), &config);
        assert_eq!(selected, 1);
    }

    #[test]
    fn test_result_to_rewards_winner() {
        let result = GameResult::Winner(PlayerId::new(1));
        let rewards = result_to_rewards(&result, 3);

        assert_eq!(rewards[PlayerId::new(0)], 0.0);
        assert_eq!(rewards[PlayerId::new(1)], 1.0);
        assert_eq!(rewards[PlayerId::new(2)], 0.0);
    }

    #[test]
    fn test_result_to_rewards_draw() {
        let result = GameResult::Draw;
        let rewards = result_to_rewards(&result, 2);

        assert_eq!(rewards[PlayerId::new(0)], 0.5);
        assert_eq!(rewards[PlayerId::new(1)], 0.5);
    }

    #[test]
    fn test_heuristic_eval() {
        let mut state = GameState::new(2, 42);
        state.public.set_player_state(PlayerId::new(0), "life", 30);
        state.public.set_player_state(PlayerId::new(1), "life", 10);

        let rewards = heuristic_eval(&state, 2);

        // Player 0 has 75% of total life
        assert!((rewards[PlayerId::new(0)] - 0.75).abs() < 0.01);
        assert!((rewards[PlayerId::new(1)] - 0.25).abs() < 0.01);
    }
}

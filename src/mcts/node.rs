//! MCTS node and edge structures.
//!
//! Uses arena-based allocation with index references (NodeId) for efficiency
//! and serializability.

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::core::{Action, PlayerId, PlayerMap};

/// Index into the MCTSTree node arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u32);

impl NodeId {
    /// Sentinel value representing no node.
    pub const NONE: NodeId = NodeId(u32::MAX);

    /// Create a new node ID.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Check if this is the NONE sentinel.
    #[inline]
    #[must_use]
    pub const fn is_none(self) -> bool {
        self.0 == u32::MAX
    }

    /// Get the raw index value.
    #[inline]
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_none() {
            write!(f, "NodeId(NONE)")
        } else {
            write!(f, "NodeId({})", self.0)
        }
    }
}

/// Edge representing an action from a parent node to a child.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Edge {
    /// The action this edge represents.
    pub action: Action,

    /// Child node (NONE if not yet expanded).
    pub child: NodeId,

    /// Visit count for this action.
    pub visits: u32,

    /// Total reward accumulated for this action (per player).
    pub total_reward: PlayerMap<f64>,

    /// Prior probability from policy network (for PUCT).
    /// Default is 1.0 for uniform prior.
    pub prior: f32,
}

impl Edge {
    /// Create a new edge with the given action.
    pub fn new(action: Action, player_count: usize) -> Self {
        Self {
            action,
            child: NodeId::NONE,
            visits: 0,
            total_reward: PlayerMap::with_value(player_count, 0.0),
            prior: 1.0,
        }
    }

    /// Create an edge with a custom prior probability.
    pub fn with_prior(action: Action, player_count: usize, prior: f32) -> Self {
        Self {
            action,
            child: NodeId::NONE,
            visits: 0,
            total_reward: PlayerMap::with_value(player_count, 0.0),
            prior,
        }
    }

    /// Get the mean reward for a player.
    #[must_use]
    pub fn mean_reward(&self, player: PlayerId) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.total_reward[player] / self.visits as f64
        }
    }

    /// Check if this edge has been expanded (child exists).
    #[must_use]
    pub fn is_expanded(&self) -> bool {
        !self.child.is_none()
    }
}

/// A node in the MCTS tree.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MCTSNode {
    /// Parent node (NONE for root).
    pub parent: NodeId,

    /// Index of the edge from parent that led to this node.
    pub parent_edge_idx: u16,

    /// Player to move at this node.
    pub to_move: PlayerId,

    /// Depth in tree (root = 0).
    pub depth: u16,

    /// Total visits to this node.
    pub visits: u32,

    /// Is this a terminal game state?
    pub is_terminal: bool,

    /// Terminal rewards (if terminal).
    pub terminal_reward: Option<PlayerMap<f64>>,

    /// Outgoing edges (available actions).
    /// SmallVec optimizes for typical branching factor < 8.
    pub edges: SmallVec<[Edge; 8]>,
}

impl MCTSNode {
    /// Create a new node.
    pub fn new(parent: NodeId, parent_edge_idx: u16, to_move: PlayerId, depth: u16) -> Self {
        Self {
            parent,
            parent_edge_idx,
            to_move,
            depth,
            visits: 0,
            is_terminal: false,
            terminal_reward: None,
            edges: SmallVec::new(),
        }
    }

    /// Create a root node.
    pub fn root(to_move: PlayerId) -> Self {
        Self::new(NodeId::NONE, 0, to_move, 0)
    }

    /// Check if all edges have been expanded.
    #[must_use]
    pub fn is_fully_expanded(&self) -> bool {
        !self.edges.is_empty() && self.edges.iter().all(|e| e.is_expanded())
    }

    /// Check if any edges are unexpanded.
    #[must_use]
    pub fn has_unexpanded(&self) -> bool {
        self.edges.iter().any(|e| !e.is_expanded())
    }

    /// Get indices of unexpanded edges.
    pub fn unexpanded_edges(&self) -> impl Iterator<Item = usize> + '_ {
        self.edges
            .iter()
            .enumerate()
            .filter(|(_, e)| !e.is_expanded())
            .map(|(i, _)| i)
    }

    /// Get the edge with the most visits.
    #[must_use]
    pub fn best_edge_by_visits(&self) -> Option<&Edge> {
        self.edges.iter().max_by_key(|e| e.visits)
    }

    /// Get the edge with the highest mean reward for a player.
    #[must_use]
    pub fn best_edge_by_reward(&self, player: PlayerId) -> Option<&Edge> {
        self.edges
            .iter()
            .max_by(|a, b| a.mean_reward(player).partial_cmp(&b.mean_reward(player)).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TemplateId;

    #[test]
    fn test_node_id() {
        let id = NodeId::new(5);
        assert_eq!(id.raw(), 5);
        assert!(!id.is_none());
        assert_eq!(format!("{}", id), "NodeId(5)");

        assert!(NodeId::NONE.is_none());
        assert_eq!(format!("{}", NodeId::NONE), "NodeId(NONE)");
    }

    #[test]
    fn test_edge_new() {
        let action = Action::new(TemplateId::new(1));
        let edge = Edge::new(action, 2);

        assert_eq!(edge.visits, 0);
        assert!(edge.child.is_none());
        assert!(!edge.is_expanded());
        assert_eq!(edge.prior, 1.0);
    }

    #[test]
    fn test_edge_mean_reward() {
        let action = Action::new(TemplateId::new(1));
        let mut edge = Edge::new(action, 2);

        // No visits = 0 reward
        assert_eq!(edge.mean_reward(PlayerId::new(0)), 0.0);

        // Add some visits and rewards
        edge.visits = 4;
        edge.total_reward[PlayerId::new(0)] = 3.0;
        edge.total_reward[PlayerId::new(1)] = 1.0;

        assert_eq!(edge.mean_reward(PlayerId::new(0)), 0.75);
        assert_eq!(edge.mean_reward(PlayerId::new(1)), 0.25);
    }

    #[test]
    fn test_node_root() {
        let node = MCTSNode::root(PlayerId::new(0));

        assert!(node.parent.is_none());
        assert_eq!(node.depth, 0);
        assert_eq!(node.to_move, PlayerId::new(0));
        assert_eq!(node.visits, 0);
        assert!(!node.is_terminal);
        assert!(node.edges.is_empty());
    }

    #[test]
    fn test_node_expansion_state() {
        let mut node = MCTSNode::root(PlayerId::new(0));

        // Empty node - has_unexpanded is false (no edges to be unexpanded)
        // is_fully_expanded is also false (we require edges to exist)
        assert!(!node.has_unexpanded());
        assert!(!node.is_fully_expanded());

        // Add unexpanded edges
        node.edges.push(Edge::new(Action::new(TemplateId::new(1)), 2));
        node.edges.push(Edge::new(Action::new(TemplateId::new(2)), 2));

        assert!(node.has_unexpanded());
        assert!(!node.is_fully_expanded());

        // Expand one
        node.edges[0].child = NodeId::new(1);

        assert!(node.has_unexpanded());
        assert!(!node.is_fully_expanded());

        // Expand all
        node.edges[1].child = NodeId::new(2);

        assert!(!node.has_unexpanded());
        assert!(node.is_fully_expanded());
    }

    #[test]
    fn test_unexpanded_edges() {
        let mut node = MCTSNode::root(PlayerId::new(0));
        node.edges.push(Edge::new(Action::new(TemplateId::new(1)), 2));
        node.edges.push(Edge::new(Action::new(TemplateId::new(2)), 2));
        node.edges.push(Edge::new(Action::new(TemplateId::new(3)), 2));

        // Expand middle one
        node.edges[1].child = NodeId::new(10);

        let unexpanded: Vec<_> = node.unexpanded_edges().collect();
        assert_eq!(unexpanded, vec![0, 2]);
    }

    #[test]
    fn test_best_edge() {
        let mut node = MCTSNode::root(PlayerId::new(0));

        let mut edge1 = Edge::new(Action::new(TemplateId::new(1)), 2);
        edge1.visits = 10;
        edge1.total_reward[PlayerId::new(0)] = 5.0;

        let mut edge2 = Edge::new(Action::new(TemplateId::new(2)), 2);
        edge2.visits = 20;
        edge2.total_reward[PlayerId::new(0)] = 8.0;

        node.edges.push(edge1);
        node.edges.push(edge2);

        // Best by visits
        let best = node.best_edge_by_visits().unwrap();
        assert_eq!(best.action.template, TemplateId::new(2));

        // Best by reward (edge1 has 0.5, edge2 has 0.4)
        let best = node.best_edge_by_reward(PlayerId::new(0)).unwrap();
        assert_eq!(best.action.template, TemplateId::new(1));
    }

    #[test]
    fn test_serialization() {
        let mut node = MCTSNode::root(PlayerId::new(1));
        node.edges.push(Edge::new(Action::new(TemplateId::new(5)), 2));
        node.visits = 100;

        let json = serde_json::to_string(&node).unwrap();
        let deserialized: MCTSNode = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.to_move, PlayerId::new(1));
        assert_eq!(deserialized.visits, 100);
        assert_eq!(deserialized.edges.len(), 1);
    }
}

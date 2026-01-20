//! Arena-based MCTS tree.
//!
//! Uses a flat `Vec<MCTSNode>` with index-based references for efficiency,
//! cache-friendliness, and serializability.

use serde::{Deserialize, Serialize};

use super::node::{MCTSNode, NodeId};
use crate::core::PlayerId;

/// Arena-based MCTS tree.
///
/// Nodes are stored in a flat vector and referenced by `NodeId` indices.
/// This avoids reference counting overhead and enables serialization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MCTSTree {
    /// All nodes in the tree.
    nodes: Vec<MCTSNode>,

    /// The root node ID (always 0 after initialization).
    root: NodeId,

    /// Number of players in the game.
    player_count: usize,
}

impl MCTSTree {
    /// Create a new tree with a root node.
    pub fn new(root_player: PlayerId, player_count: usize) -> Self {
        let mut tree = Self {
            nodes: Vec::with_capacity(1024),
            root: NodeId::new(0),
            player_count,
        };
        tree.nodes.push(MCTSNode::root(root_player));
        tree
    }

    /// Create a tree with custom initial capacity.
    pub fn with_capacity(root_player: PlayerId, player_count: usize, capacity: usize) -> Self {
        let mut tree = Self {
            nodes: Vec::with_capacity(capacity),
            root: NodeId::new(0),
            player_count,
        };
        tree.nodes.push(MCTSNode::root(root_player));
        tree
    }

    /// Get the root node ID.
    #[inline]
    #[must_use]
    pub fn root(&self) -> NodeId {
        self.root
    }

    /// Get a node by ID.
    #[inline]
    #[must_use]
    pub fn get(&self, id: NodeId) -> &MCTSNode {
        &self.nodes[id.0 as usize]
    }

    /// Get a mutable node by ID.
    #[inline]
    pub fn get_mut(&mut self, id: NodeId) -> &mut MCTSNode {
        &mut self.nodes[id.0 as usize]
    }

    /// Allocate a new node, returning its ID.
    pub fn alloc(&mut self, node: MCTSNode) -> NodeId {
        let id = NodeId::new(self.nodes.len() as u32);
        self.nodes.push(node);
        id
    }

    /// Number of nodes in the tree.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the tree is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Player count for this tree.
    #[must_use]
    pub fn player_count(&self) -> usize {
        self.player_count
    }

    /// Get statistics about the tree.
    #[must_use]
    pub fn stats(&self) -> TreeStats {
        let max_depth = self.nodes.iter().map(|n| n.depth).max().unwrap_or(0);
        let terminal_count = self.nodes.iter().filter(|n| n.is_terminal).count();
        let total_edges: usize = self.nodes.iter().map(|n| n.edges.len()).sum();
        let expanded_edges: usize = self.nodes.iter()
            .flat_map(|n| n.edges.iter())
            .filter(|e| e.is_expanded())
            .count();

        TreeStats {
            node_count: self.nodes.len(),
            max_depth,
            terminal_count,
            total_edges,
            expanded_edges,
        }
    }

    /// Clear the tree and reset with a new root.
    pub fn reset(&mut self, root_player: PlayerId) {
        self.nodes.clear();
        self.nodes.push(MCTSNode::root(root_player));
        self.root = NodeId::new(0);
    }

    /// Get the root node.
    #[must_use]
    pub fn root_node(&self) -> &MCTSNode {
        self.get(self.root)
    }

    /// Get the root node mutably.
    pub fn root_node_mut(&mut self) -> &mut MCTSNode {
        self.get_mut(self.root)
    }

    /// Iterate over all nodes.
    pub fn iter(&self) -> impl Iterator<Item = (NodeId, &MCTSNode)> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (NodeId::new(i as u32), n))
    }
}

/// Statistics about the MCTS tree.
#[derive(Clone, Debug, Default)]
pub struct TreeStats {
    /// Total number of nodes.
    pub node_count: usize,

    /// Maximum depth reached.
    pub max_depth: u16,

    /// Number of terminal nodes.
    pub terminal_count: usize,

    /// Total number of edges (actions).
    pub total_edges: usize,

    /// Number of expanded edges (with children).
    pub expanded_edges: usize,
}

impl TreeStats {
    /// Get the branching factor (average edges per node).
    #[must_use]
    pub fn branching_factor(&self) -> f64 {
        if self.node_count == 0 {
            0.0
        } else {
            self.total_edges as f64 / self.node_count as f64
        }
    }

    /// Get the expansion ratio (expanded edges / total edges).
    #[must_use]
    pub fn expansion_ratio(&self) -> f64 {
        if self.total_edges == 0 {
            0.0
        } else {
            self.expanded_edges as f64 / self.total_edges as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Action, TemplateId};
    use crate::mcts::node::Edge;

    #[test]
    fn test_tree_new() {
        let tree = MCTSTree::new(PlayerId::new(0), 2);

        assert_eq!(tree.len(), 1);
        assert!(!tree.is_empty());
        assert_eq!(tree.player_count(), 2);
        assert_eq!(tree.root(), NodeId::new(0));
    }

    #[test]
    fn test_tree_alloc() {
        let mut tree = MCTSTree::new(PlayerId::new(0), 2);

        let child = MCTSNode::new(NodeId::new(0), 0, PlayerId::new(1), 1);
        let child_id = tree.alloc(child);

        assert_eq!(child_id, NodeId::new(1));
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.get(child_id).to_move, PlayerId::new(1));
    }

    #[test]
    fn test_tree_get_mut() {
        let mut tree = MCTSTree::new(PlayerId::new(0), 2);

        tree.get_mut(tree.root()).visits = 100;

        assert_eq!(tree.get(tree.root()).visits, 100);
    }

    #[test]
    fn test_tree_reset() {
        let mut tree = MCTSTree::new(PlayerId::new(0), 2);

        // Add some nodes
        tree.alloc(MCTSNode::new(NodeId::new(0), 0, PlayerId::new(1), 1));
        tree.alloc(MCTSNode::new(NodeId::new(1), 0, PlayerId::new(0), 2));

        assert_eq!(tree.len(), 3);

        // Reset
        tree.reset(PlayerId::new(1));

        assert_eq!(tree.len(), 1);
        assert_eq!(tree.root_node().to_move, PlayerId::new(1));
    }

    #[test]
    fn test_tree_stats() {
        let mut tree = MCTSTree::new(PlayerId::new(0), 2);

        // Add edges to root
        let root = tree.root();
        tree.get_mut(root).edges.push(Edge::new(Action::new(TemplateId::new(1)), 2));
        tree.get_mut(root).edges.push(Edge::new(Action::new(TemplateId::new(2)), 2));

        // Expand one edge
        let child = MCTSNode::new(root, 0, PlayerId::new(1), 1);
        let child_id = tree.alloc(child);
        tree.get_mut(root).edges[0].child = child_id;

        // Mark child as terminal
        tree.get_mut(child_id).is_terminal = true;

        let stats = tree.stats();

        assert_eq!(stats.node_count, 2);
        assert_eq!(stats.max_depth, 1);
        assert_eq!(stats.terminal_count, 1);
        assert_eq!(stats.total_edges, 2);
        assert_eq!(stats.expanded_edges, 1);
        assert_eq!(stats.expansion_ratio(), 0.5);
    }

    #[test]
    fn test_tree_iter() {
        let mut tree = MCTSTree::new(PlayerId::new(0), 2);
        tree.alloc(MCTSNode::new(NodeId::new(0), 0, PlayerId::new(1), 1));

        let nodes: Vec<_> = tree.iter().collect();

        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].0, NodeId::new(0));
        assert_eq!(nodes[1].0, NodeId::new(1));
    }

    #[test]
    fn test_tree_serialization() {
        let mut tree = MCTSTree::new(PlayerId::new(0), 2);
        tree.get_mut(tree.root()).visits = 50;
        tree.alloc(MCTSNode::new(NodeId::new(0), 0, PlayerId::new(1), 1));

        let json = serde_json::to_string(&tree).unwrap();
        let deserialized: MCTSTree = serde_json::from_str(&json).unwrap();

        assert_eq!(tree.len(), deserialized.len());
        assert_eq!(tree.player_count(), deserialized.player_count());
        assert_eq!(tree.root_node().visits, deserialized.root_node().visits);
    }
}

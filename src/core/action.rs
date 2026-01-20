//! Action representation: template + entity pointers.
//!
//! Actions are compositional: a template (the "verb") plus entity pointers
//! (the "nouns"). For example:
//! - "Pass" = template only, no pointers
//! - "Play card X" = template + 1 pointer (the card)
//! - "Attack with X targeting Y" = template + 2 pointers
//!
//! Games define their templates via `TemplateConfig`. The engine doesn't
//! interpret templates - it just stores and compares them.

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use super::config::TemplateId;
use super::entity::EntityId;
use super::player::PlayerId;

/// A complete game action.
///
/// Actions consist of:
/// - A template ID (the type of action)
/// - Zero or more entity pointers (targets, sources, etc.)
///
/// ## Example
///
/// ```
/// use rust_ccg::core::{Action, TemplateId, EntityId};
///
/// // "Pass" action - no pointers
/// let pass = Action::new(TemplateId::new(0));
///
/// // "Play card" action - one pointer (the card)
/// let play = Action::with_pointers(TemplateId::new(1), &[EntityId(5)]);
///
/// // "Cast spell on target" - two pointers (spell, target)
/// let cast = Action::with_pointers(TemplateId::new(2), &[EntityId(10), EntityId(0)]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Action {
    /// The action template (type of action).
    pub template: TemplateId,

    /// Entity pointers for this action.
    /// SmallVec optimizes for 0-3 pointers (common case) without heap allocation.
    pub pointers: SmallVec<[EntityId; 3]>,
}

impl Action {
    /// Create an action with no pointers.
    #[must_use]
    pub fn new(template: TemplateId) -> Self {
        Self {
            template,
            pointers: SmallVec::new(),
        }
    }

    /// Create an action with the given pointers.
    #[must_use]
    pub fn with_pointers(template: TemplateId, pointers: &[EntityId]) -> Self {
        Self {
            template,
            pointers: SmallVec::from_slice(pointers),
        }
    }

    /// Add a pointer to this action.
    pub fn push_pointer(&mut self, entity: EntityId) {
        self.pointers.push(entity);
    }

    /// Get the number of pointers.
    #[must_use]
    pub fn pointer_count(&self) -> usize {
        self.pointers.len()
    }

    /// Check if this action has no pointers.
    #[must_use]
    pub fn is_no_arg(&self) -> bool {
        self.pointers.is_empty()
    }
}

/// A recorded action with metadata for history tracking.
///
/// Used for:
/// - Action history in MCTS (opponent consistency)
/// - Replay/debugging
/// - Training data
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionRecord {
    /// The player who took this action.
    pub player: PlayerId,

    /// The action taken.
    pub action: Action,

    /// Turn number when action was taken.
    pub turn: u32,

    /// Sequence number within the turn (for ordering).
    pub sequence: u32,
}

impl ActionRecord {
    /// Create a new action record.
    #[must_use]
    pub fn new(player: PlayerId, action: Action, turn: u32, sequence: u32) -> Self {
        Self {
            player,
            action,
            turn,
            sequence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_no_pointers() {
        let action = Action::new(TemplateId::new(0));

        assert_eq!(action.template, TemplateId::new(0));
        assert!(action.is_no_arg());
        assert_eq!(action.pointer_count(), 0);
    }

    #[test]
    fn test_action_with_pointers() {
        let action = Action::with_pointers(
            TemplateId::new(1),
            &[EntityId(5), EntityId(10)],
        );

        assert_eq!(action.template, TemplateId::new(1));
        assert!(!action.is_no_arg());
        assert_eq!(action.pointer_count(), 2);
        assert_eq!(action.pointers[0], EntityId(5));
        assert_eq!(action.pointers[1], EntityId(10));
    }

    #[test]
    fn test_action_push_pointer() {
        let mut action = Action::new(TemplateId::new(2));
        action.push_pointer(EntityId(1));
        action.push_pointer(EntityId(2));

        assert_eq!(action.pointer_count(), 2);
        assert_eq!(action.pointers[0], EntityId(1));
        assert_eq!(action.pointers[1], EntityId(2));
    }

    #[test]
    fn test_action_equality() {
        let a1 = Action::with_pointers(TemplateId::new(1), &[EntityId(5)]);
        let a2 = Action::with_pointers(TemplateId::new(1), &[EntityId(5)]);
        let a3 = Action::with_pointers(TemplateId::new(1), &[EntityId(6)]);
        let a4 = Action::with_pointers(TemplateId::new(2), &[EntityId(5)]);

        assert_eq!(a1, a2);
        assert_ne!(a1, a3);
        assert_ne!(a1, a4);
    }

    #[test]
    fn test_action_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let hash = |a: &Action| {
            let mut h = DefaultHasher::new();
            a.hash(&mut h);
            h.finish()
        };

        let a1 = Action::with_pointers(TemplateId::new(1), &[EntityId(5)]);
        let a2 = Action::with_pointers(TemplateId::new(1), &[EntityId(5)]);
        let a3 = Action::with_pointers(TemplateId::new(1), &[EntityId(6)]);

        assert_eq!(hash(&a1), hash(&a2));
        assert_ne!(hash(&a1), hash(&a3));
    }

    #[test]
    fn test_action_record() {
        let action = Action::with_pointers(TemplateId::new(1), &[EntityId(5)]);
        let record = ActionRecord::new(PlayerId::new(0), action.clone(), 3, 5);

        assert_eq!(record.player, PlayerId::new(0));
        assert_eq!(record.action, action);
        assert_eq!(record.turn, 3);
        assert_eq!(record.sequence, 5);
    }

    #[test]
    fn test_action_serialization() {
        let action = Action::with_pointers(TemplateId::new(1), &[EntityId(5), EntityId(10)]);
        let json = serde_json::to_string(&action).unwrap();
        let deserialized: Action = serde_json::from_str(&json).unwrap();

        assert_eq!(action, deserialized);
    }

    #[test]
    fn test_action_record_serialization() {
        let action = Action::with_pointers(TemplateId::new(1), &[EntityId(5)]);
        let record = ActionRecord::new(PlayerId::new(1), action, 2, 3);

        let json = serde_json::to_string(&record).unwrap();
        let deserialized: ActionRecord = serde_json::from_str(&json).unwrap();

        assert_eq!(record, deserialized);
    }
}

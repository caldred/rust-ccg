//! Priority-based stack resolution (MTG-style).
//!
//! Effects are pushed onto a stack and resolve in LIFO order.
//! Players pass priority in turn order; when all pass, the top
//! of the stack resolves.

use serde::{Deserialize, Serialize};

use crate::core::{Action, GameState, PlayerId};
use crate::effects::{EffectBatch, EffectResolver, ResolverContext};
use crate::triggers::{GameEvent, TriggeredEffect, TriggerId};

use super::{ResolutionStatus, ResolutionSystem};

/// Unique identifier for a stack entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StackEntryId(pub u32);

impl StackEntryId {
    /// Create a new stack entry ID.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw ID value.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for StackEntryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StackEntry({})", self.0)
    }
}

/// What caused a stack entry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum StackSource {
    /// An action being resolved.
    Action(Action),

    /// A triggered ability.
    Triggered {
        trigger_id: TriggerId,
        event: GameEvent,
    },

    /// A response/instant effect.
    Response {
        description: String,
    },
}

/// An entry on the stack.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StackEntry {
    /// Unique identifier for this entry.
    pub id: StackEntryId,

    /// The effects to resolve.
    pub effects: EffectBatch,

    /// Who controls this entry (makes choices, gets priority after).
    pub controller: PlayerId,

    /// What caused this entry.
    pub source: StackSource,
}

/// Priority-based stack resolution system.
///
/// Implements MTG-style stack resolution:
/// 1. Effects are pushed onto a stack
/// 2. Active player gets priority
/// 3. Players pass priority in turn order
/// 4. When all players pass in sequence, top of stack resolves
/// 5. After resolution, active player gets priority again
/// 6. Repeat until stack is empty
///
/// ## N-Player Support
///
/// Priority passes in player order (0 → 1 → 2 → ... → 0).
/// All players must pass consecutively for resolution to occur.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PriorityStack {
    /// The stack (index 0 = bottom, last = top).
    entries: Vec<StackEntry>,

    /// Triggered effects waiting to be added to stack.
    pending_triggers: Vec<TriggeredEffect>,

    /// Player who currently has priority.
    current_priority: PlayerId,

    /// Players who have passed since last action/resolution.
    /// When this equals player_count, we resolve.
    consecutive_passes: usize,

    /// Total player count.
    player_count: usize,

    /// Next stack entry ID.
    next_id: u32,
}

impl PriorityStack {
    /// Create a new priority stack for the given number of players.
    ///
    /// Priority starts with player 0.
    pub fn new(player_count: usize) -> Self {
        Self {
            entries: Vec::new(),
            pending_triggers: Vec::new(),
            current_priority: PlayerId::new(0),
            consecutive_passes: 0,
            player_count,
            next_id: 0,
        }
    }

    /// Create with a specific starting priority player.
    pub fn with_priority(player_count: usize, starting_priority: PlayerId) -> Self {
        Self {
            entries: Vec::new(),
            pending_triggers: Vec::new(),
            current_priority: starting_priority,
            consecutive_passes: 0,
            player_count,
            next_id: 0,
        }
    }

    /// Get the number of entries on the stack.
    #[must_use]
    pub fn stack_size(&self) -> usize {
        self.entries.len()
    }

    /// Get the number of pending triggered effects.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending_triggers.len()
    }

    /// Peek at the top of the stack without removing it.
    #[must_use]
    pub fn peek_top(&self) -> Option<&StackEntry> {
        self.entries.last()
    }

    /// Get all entries on the stack (bottom to top).
    #[must_use]
    pub fn entries(&self) -> &[StackEntry] {
        &self.entries
    }

    /// Player passes priority.
    ///
    /// Returns `true` if all players have now passed (stack should resolve).
    pub fn pass(&mut self, player: PlayerId) -> bool {
        if player != self.current_priority {
            return false; // Wrong player
        }

        self.consecutive_passes += 1;

        if self.consecutive_passes >= self.player_count {
            // All players passed - ready to resolve
            true
        } else {
            // Advance to next player
            self.advance_priority();
            false
        }
    }

    /// Player responds by adding effects to the stack.
    ///
    /// This resets the pass counter and gives the responding player priority.
    pub fn respond(&mut self, effects: EffectBatch, controller: PlayerId, description: String) {
        let id = StackEntryId::new(self.next_id);
        self.next_id += 1;

        self.entries.push(StackEntry {
            id,
            effects,
            controller,
            source: StackSource::Response { description },
        });

        // Reset passes and give priority to the responding player
        self.consecutive_passes = 0;
        self.current_priority = controller;
    }

    /// Flush pending triggers to the stack.
    ///
    /// Triggers are sorted by priority (higher first) before being added.
    /// Call this after resolving a stack entry to add any triggered effects.
    pub fn flush_triggers(&mut self) {
        if self.pending_triggers.is_empty() {
            return;
        }

        // Sort by priority (already sorted by TriggerRegistry, but ensure stable order)
        // Higher priority triggers go on stack last (resolve first due to LIFO)
        self.pending_triggers.sort_by(|a, b| {
            // Sort by trigger_id for stability (TriggerRegistry already sorted by priority)
            a.trigger_id.0.cmp(&b.trigger_id.0)
        });

        // Add to stack in order (first trigger = bottom of new batch)
        for triggered in self.pending_triggers.drain(..) {
            let id = StackEntryId::new(self.next_id);
            self.next_id += 1;

            let mut batch = EffectBatch::new();
            for effect in triggered.effects {
                if let Some(controller) = triggered.controller {
                    batch.add_player(effect, controller);
                } else {
                    batch.add_zone(effect);
                }
            }

            let controller = triggered.controller.unwrap_or(PlayerId::new(0));

            self.entries.push(StackEntry {
                id,
                effects: batch,
                controller,
                source: StackSource::Triggered {
                    trigger_id: triggered.trigger_id,
                    event: triggered.triggering_event,
                },
            });
        }

        // Reset passes after adding triggers
        self.consecutive_passes = 0;
    }

    /// Resolve the top of the stack.
    ///
    /// Returns the resolved entry, or `None` if stack is empty.
    fn resolve_top(&mut self, state: &mut GameState, context: &ResolverContext) -> Option<StackEntry> {
        let entry = self.entries.pop()?;
        let _results = EffectResolver::resolve_batch(state, &entry.effects, context);

        // Reset passes and give priority to active player (or entry controller)
        self.consecutive_passes = 0;
        // In MTG, active player gets priority after resolution
        // We'll use the entry's controller for now; games can override
        self.current_priority = entry.controller;

        Some(entry)
    }

    /// Advance priority to the next player.
    fn advance_priority(&mut self) {
        let next = (self.current_priority.0 as usize + 1) % self.player_count;
        self.current_priority = PlayerId::new(next as u8);
    }

    /// Set the priority player explicitly.
    pub fn set_priority(&mut self, player: PlayerId) {
        self.current_priority = player;
        self.consecutive_passes = 0;
    }
}

impl ResolutionSystem for PriorityStack {
    fn queue_action(
        &mut self,
        source_action: Action,
        effects: EffectBatch,
        controller: PlayerId,
    ) {
        if effects.is_empty() {
            return;
        }

        let id = StackEntryId::new(self.next_id);
        self.next_id += 1;

        self.entries.push(StackEntry {
            id,
            effects,
            controller,
            source: StackSource::Action(source_action),
        });

        // Reset passes and give priority to controller
        self.consecutive_passes = 0;
        self.current_priority = controller;
    }

    fn queue_triggered(&mut self, triggered: TriggeredEffect) {
        if !triggered.effects.is_empty() {
            self.pending_triggers.push(triggered);
        }
    }

    fn process(&mut self, state: &mut GameState, context: &ResolverContext) -> ResolutionStatus {
        // First, flush any pending triggers to the stack
        if !self.pending_triggers.is_empty() {
            self.flush_triggers();
        }

        // If stack is empty, we're done
        if self.entries.is_empty() {
            return ResolutionStatus::Complete;
        }

        // Check if all players have passed
        if self.consecutive_passes >= self.player_count {
            // Resolve top of stack
            self.resolve_top(state, context);

            // Check if more entries remain
            if self.entries.is_empty() && self.pending_triggers.is_empty() {
                return ResolutionStatus::Complete;
            }

            // More to process
            return ResolutionStatus::Processing;
        }

        // Waiting for current player to pass or respond
        ResolutionStatus::WaitingForPriority(self.current_priority)
    }

    fn is_complete(&self) -> bool {
        self.entries.is_empty() && self.pending_triggers.is_empty()
    }

    fn priority_player(&self) -> Option<PlayerId> {
        if self.entries.is_empty() && self.pending_triggers.is_empty() {
            None
        } else {
            Some(self.current_priority)
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.pending_triggers.clear();
        self.consecutive_passes = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TemplateId;
    use crate::effects::Effect;

    #[test]
    fn test_stack_entry_id() {
        let id = StackEntryId::new(5);
        assert_eq!(id.raw(), 5);
        assert_eq!(format!("{}", id), "StackEntry(5)");
    }

    #[test]
    fn test_priority_stack_new() {
        let stack = PriorityStack::new(2);
        assert!(stack.is_complete());
        assert_eq!(stack.stack_size(), 0);
        assert_eq!(stack.priority_player(), None);
    }

    #[test]
    fn test_queue_action() {
        let mut stack = PriorityStack::new(2);

        let mut batch = EffectBatch::new();
        batch.add_player(Effect::damage(3), PlayerId::new(1));

        let action = Action::new(TemplateId::new(1));
        stack.queue_action(action, batch, PlayerId::new(0));

        assert_eq!(stack.stack_size(), 1);
        assert!(!stack.is_complete());
        assert_eq!(stack.priority_player(), Some(PlayerId::new(0)));
    }

    #[test]
    fn test_pass_priority_two_player() {
        let mut stack = PriorityStack::new(2);

        let mut batch = EffectBatch::new();
        batch.add_player(Effect::damage(3), PlayerId::new(1));
        stack.queue_action(Action::new(TemplateId::new(1)), batch, PlayerId::new(0));

        // Player 0 has priority
        assert_eq!(stack.priority_player(), Some(PlayerId::new(0)));

        // Player 0 passes
        let all_passed = stack.pass(PlayerId::new(0));
        assert!(!all_passed);
        assert_eq!(stack.priority_player(), Some(PlayerId::new(1)));

        // Player 1 passes - all have passed
        let all_passed = stack.pass(PlayerId::new(1));
        assert!(all_passed);
    }

    #[test]
    fn test_pass_priority_four_player() {
        let mut stack = PriorityStack::new(4);

        let mut batch = EffectBatch::new();
        batch.add_player(Effect::damage(3), PlayerId::new(1));
        stack.queue_action(Action::new(TemplateId::new(1)), batch, PlayerId::new(0));

        // All 4 players must pass
        assert!(!stack.pass(PlayerId::new(0)));
        assert_eq!(stack.priority_player(), Some(PlayerId::new(1)));

        assert!(!stack.pass(PlayerId::new(1)));
        assert_eq!(stack.priority_player(), Some(PlayerId::new(2)));

        assert!(!stack.pass(PlayerId::new(2)));
        assert_eq!(stack.priority_player(), Some(PlayerId::new(3)));

        // Fourth pass triggers resolution
        assert!(stack.pass(PlayerId::new(3)));
    }

    #[test]
    fn test_respond_resets_passes() {
        let mut stack = PriorityStack::new(2);

        let mut batch = EffectBatch::new();
        batch.add_player(Effect::damage(3), PlayerId::new(1));
        stack.queue_action(Action::new(TemplateId::new(1)), batch, PlayerId::new(0));

        // Player 0 passes
        stack.pass(PlayerId::new(0));
        assert_eq!(stack.priority_player(), Some(PlayerId::new(1)));

        // Player 1 responds instead of passing
        let mut response = EffectBatch::new();
        response.add_player(Effect::heal(2), PlayerId::new(1));
        stack.respond(response, PlayerId::new(1), "Counter".to_string());

        // Stack now has 2 entries, passes reset
        assert_eq!(stack.stack_size(), 2);
        assert_eq!(stack.priority_player(), Some(PlayerId::new(1)));

        // Now both must pass again
        assert!(!stack.pass(PlayerId::new(1)));
        assert!(stack.pass(PlayerId::new(0)));
    }

    #[test]
    fn test_lifo_resolution() {
        let mut stack = PriorityStack::new(2);
        let mut state = GameState::new(2, 42);
        state.public.set_player_state(PlayerId::new(1), "life", 20);

        // Push first effect (3 damage)
        let mut batch1 = EffectBatch::new();
        batch1.add_player(Effect::damage(3), PlayerId::new(1));
        stack.queue_action(Action::new(TemplateId::new(1)), batch1, PlayerId::new(0));

        // Push second effect (5 damage) - this should resolve first (LIFO)
        let mut batch2 = EffectBatch::new();
        batch2.add_player(Effect::damage(5), PlayerId::new(1));
        stack.respond(batch2, PlayerId::new(0), "Response".to_string());

        assert_eq!(stack.stack_size(), 2);

        let context = ResolverContext::simple(2);

        // Both pass
        stack.pass(PlayerId::new(0));
        stack.pass(PlayerId::new(1));

        // Process - should resolve top (5 damage)
        let status = stack.process(&mut state, &context);
        assert_eq!(status, ResolutionStatus::Processing);
        assert_eq!(stack.stack_size(), 1);

        // Life should be 20 - 5 = 15
        assert_eq!(state.public.get_player_state(PlayerId::new(1), "life", 0), 15);

        // Both pass again
        stack.pass(PlayerId::new(0));
        stack.pass(PlayerId::new(1));

        // Process - should resolve remaining (3 damage)
        let status = stack.process(&mut state, &context);
        assert_eq!(status, ResolutionStatus::Complete);
        assert_eq!(stack.stack_size(), 0);

        // Life should be 15 - 3 = 12
        assert_eq!(state.public.get_player_state(PlayerId::new(1), "life", 0), 12);
    }

    #[test]
    fn test_process_waiting_for_priority() {
        let mut stack = PriorityStack::new(2);
        let mut state = GameState::new(2, 42);

        let mut batch = EffectBatch::new();
        batch.add_player(Effect::damage(3), PlayerId::new(1));
        stack.queue_action(Action::new(TemplateId::new(1)), batch, PlayerId::new(0));

        let context = ResolverContext::simple(2);

        // Process without passes - should wait
        let status = stack.process(&mut state, &context);
        assert_eq!(status, ResolutionStatus::WaitingForPriority(PlayerId::new(0)));
    }

    #[test]
    fn test_clear() {
        let mut stack = PriorityStack::new(2);

        let mut batch = EffectBatch::new();
        batch.add_player(Effect::damage(3), PlayerId::new(1));
        stack.queue_action(Action::new(TemplateId::new(1)), batch, PlayerId::new(0));

        assert!(!stack.is_complete());

        stack.clear();

        assert!(stack.is_complete());
        assert_eq!(stack.stack_size(), 0);
    }

    #[test]
    fn test_empty_batch_not_queued() {
        let mut stack = PriorityStack::new(2);

        let batch = EffectBatch::new(); // Empty
        stack.queue_action(Action::new(TemplateId::new(1)), batch, PlayerId::new(0));

        assert!(stack.is_complete());
        assert_eq!(stack.stack_size(), 0);
    }
}

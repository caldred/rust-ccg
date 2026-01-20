//! Immediate resolution system (Hearthstone-style).
//!
//! Effects resolve immediately as they're queued. No priority passing,
//! no stack - effects just happen in order.

use serde::{Deserialize, Serialize};

use crate::core::{Action, GameState, PlayerId};
use crate::effects::{EffectBatch, EffectResolver, ResolverContext};
use crate::triggers::TriggeredEffect;

use super::{ResolutionStatus, ResolutionSystem};

/// A pending effect waiting to be resolved.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct PendingEffect {
    /// The effects to resolve.
    effects: EffectBatch,
    /// Who controls this effect (for targeting decisions).
    controller: PlayerId,
}

/// Immediate resolution system.
///
/// Effects resolve immediately as they're queued. This is the simplest
/// resolution model, suitable for games like Hearthstone where effects
/// just happen without a response window.
///
/// ## Behavior
///
/// - `queue_action()`: Adds effects to pending queue
/// - `queue_triggered()`: Adds triggered effects to pending queue
/// - `process()`: Resolves all pending effects immediately, returns `Complete`
/// - `priority_player()`: Always returns `None` (no priority in immediate mode)
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ImmediateResolution {
    /// Pending effects to resolve.
    pending: Vec<PendingEffect>,
}

impl ImmediateResolution {
    /// Create a new immediate resolution system.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the number of pending effects.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

impl ResolutionSystem for ImmediateResolution {
    fn queue_action(
        &mut self,
        _source_action: Action,
        effects: EffectBatch,
        controller: PlayerId,
    ) {
        if !effects.is_empty() {
            self.pending.push(PendingEffect { effects, controller });
        }
    }

    fn queue_triggered(&mut self, triggered: TriggeredEffect) {
        if !triggered.effects.is_empty() {
            let mut batch = EffectBatch::new();
            for effect in triggered.effects {
                // For triggered effects, apply to controller by default
                // Games can override targeting via the effect itself
                if let Some(controller) = triggered.controller {
                    batch.add_player(effect, controller);
                } else {
                    // No controller, add as zone effect (no target)
                    batch.add_zone(effect);
                }
            }
            // Use controller if available, otherwise default to player 0
            // (consistent with PriorityStack behavior)
            let controller = triggered.controller.unwrap_or(PlayerId::new(0));
            self.pending.push(PendingEffect {
                effects: batch,
                controller,
            });
        }
    }

    fn process(&mut self, state: &mut GameState, context: &ResolverContext) -> ResolutionStatus {
        // Resolve all pending effects immediately
        while let Some(pending) = self.pending.pop() {
            let _results = EffectResolver::resolve_batch(state, &pending.effects, context);
            // In immediate mode, we don't stop for anything - just keep resolving
        }

        ResolutionStatus::Complete
    }

    fn is_complete(&self) -> bool {
        self.pending.is_empty()
    }

    fn priority_player(&self) -> Option<PlayerId> {
        // No priority in immediate mode
        None
    }

    fn clear(&mut self) {
        self.pending.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::Effect;

    #[test]
    fn test_immediate_new() {
        let resolver = ImmediateResolution::new();
        assert!(resolver.is_complete());
        assert_eq!(resolver.pending_count(), 0);
        assert_eq!(resolver.priority_player(), None);
    }

    #[test]
    fn test_queue_action() {
        let mut resolver = ImmediateResolution::new();

        let mut batch = EffectBatch::new();
        batch.add_player(Effect::damage(3), PlayerId::new(1));

        let action = Action::new(crate::core::TemplateId::new(1));
        resolver.queue_action(action, batch, PlayerId::new(0));

        assert_eq!(resolver.pending_count(), 1);
        assert!(!resolver.is_complete());
    }

    #[test]
    fn test_queue_empty_batch() {
        let mut resolver = ImmediateResolution::new();

        let batch = EffectBatch::new(); // Empty
        let action = Action::new(crate::core::TemplateId::new(1));
        resolver.queue_action(action, batch, PlayerId::new(0));

        // Empty batches are not queued
        assert_eq!(resolver.pending_count(), 0);
        assert!(resolver.is_complete());
    }

    #[test]
    fn test_process_resolves_all() {
        let mut resolver = ImmediateResolution::new();

        // Queue multiple effects
        for i in 0..3 {
            let mut batch = EffectBatch::new();
            batch.add_player(Effect::damage(i + 1), PlayerId::new(1));
            let action = Action::new(crate::core::TemplateId::new(1));
            resolver.queue_action(action, batch, PlayerId::new(0));
        }

        assert_eq!(resolver.pending_count(), 3);

        let mut state = GameState::new(2, 42);
        let context = ResolverContext::simple(2);
        let status = resolver.process(&mut state, &context);

        assert_eq!(status, ResolutionStatus::Complete);
        assert!(resolver.is_complete());
        assert_eq!(resolver.pending_count(), 0);
    }

    #[test]
    fn test_clear() {
        let mut resolver = ImmediateResolution::new();

        let mut batch = EffectBatch::new();
        batch.add_player(Effect::damage(3), PlayerId::new(1));
        let action = Action::new(crate::core::TemplateId::new(1));
        resolver.queue_action(action, batch, PlayerId::new(0));

        assert!(!resolver.is_complete());

        resolver.clear();

        assert!(resolver.is_complete());
        assert_eq!(resolver.pending_count(), 0);
    }

    #[test]
    fn test_damage_actually_applied() {
        let mut resolver = ImmediateResolution::new();
        let mut state = GameState::new(2, 42);

        // Set initial life
        state.public.set_player_state(PlayerId::new(1), "life", 20);

        // Queue damage
        let mut batch = EffectBatch::new();
        batch.add_player(Effect::damage(5), PlayerId::new(1));
        let action = Action::new(crate::core::TemplateId::new(1));
        resolver.queue_action(action, batch, PlayerId::new(0));

        let context = ResolverContext::simple(2);
        resolver.process(&mut state, &context);

        // Damage should be applied (life reduced by 5)
        let life = state.public.get_player_state(PlayerId::new(1), "life", 0);
        assert_eq!(life, 15);
    }
}

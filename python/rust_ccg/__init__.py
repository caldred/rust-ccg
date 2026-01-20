"""rust-ccg: A card game engine for AlphaZero-style training.

This package provides Python bindings for the Rust card game engine,
enabling fast MCTS self-play with neural network integration.

Quick Start:
    >>> import rust_ccg as ccg
    >>> config = ccg.SelfPlayConfig(mcts_iterations=100)
    >>> worker = ccg.SimpleGameWorker(player_count=2, config=config)
    >>> trajectory = worker.play_game(seed=42)
    >>> samples = trajectory.to_training_samples()

Classes:
    Core Types:
        PlayerId: Player identifier (0-255)
        Action: Game action with template ID and targets
        TemplateId: Action template identifier

    Neural Network:
        EncodedState: Tensor representation of game state
        PolicyValueNetwork: Wrapper for Python neural networks
        UniformPolicy: Baseline uniform random policy
        SimpleEncoder: Basic state encoder

    Training:
        Step: Single game step with state, action, and MCTS policy
        Trajectory: Complete game record with outcome
        TrainingSample: Training example (state, policy, value)
        ExperienceBuffer: FIFO buffer for trajectories

    Self-Play:
        SelfPlayConfig: Configuration for MCTS self-play
        SimpleGameWorker: Self-play worker for SimpleGame

    Games:
        SimpleGame: Simple card game for testing
"""

# Import all classes from the Rust extension
from .rust_ccg import (
    # Core types
    PlayerId,
    Action,
    TemplateId,
    # Neural network
    EncodedState,
    PolicyValueNetwork,
    UniformPolicy,
    SimpleEncoder,
    # Training
    Step,
    Trajectory,
    TrainingSample,
    ExperienceBuffer,
    # Self-play
    SelfPlayConfig,
    SimpleGameWorker,
    # Games
    SimpleGame,
)

__all__ = [
    # Core types
    "PlayerId",
    "Action",
    "TemplateId",
    # Neural network
    "EncodedState",
    "PolicyValueNetwork",
    "UniformPolicy",
    "SimpleEncoder",
    # Training
    "Step",
    "Trajectory",
    "TrainingSample",
    "ExperienceBuffer",
    # Self-play
    "SelfPlayConfig",
    "SimpleGameWorker",
    # Games
    "SimpleGame",
]

__version__ = "0.1.0"

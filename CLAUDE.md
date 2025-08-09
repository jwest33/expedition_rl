# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Setup
```bash
# Install package in editable mode with dependencies
pip install -e .

# Optional: Install training dependencies
pip install stable-baselines3
```

### Running Code
```bash
# Run random agent demo
python scripts/run_random_agent.py

# Launch interactive GUI game
python scripts/play_gui.py

# Train PPO agent (requires stable-baselines3)
python scripts/train_ppo_simple.py  # Simplified version without TensorBoard
# or
python scripts/train_ppo.py  # Original with TensorBoard (may have compatibility issues)
```

### Testing
```bash
# Run tests with pytest
python -m pytest tests/ -v

# Run specific test file
python tests/smoke_test.py
```

## Architecture Overview

This is a reinforcement learning environment for experimenting with PPO strategies under highly variable starting conditions and temporal dependencies. The codebase follows a modular design:

### Core Components

**expedition_rl/env.py**: Main gymnasium environment implementation (`ExpeditionEnv`)
- Handles agent position, goal, resources (food, fuel, health, morale)
- Implements stochastic events and weather simulation
- Action space: 8 discrete actions (move, stay, forage, shortcut, repair)
- Observation space: normalized features including position, resources, weather state

**expedition_rl/configs.py**: Configuration dataclass (`ExpeditionConfig`)
- Defines all hyperparameters and environment settings
- Controls starting condition variability ranges
- Sets reward scales, penalties, and event rates

**expedition_rl/simulators.py**: Weather simulation and event sampling
- `WeatherSim`: Markov chain weather transitions (Clear/Wind/Storm)
- `make_terrain_risk`: Generates smoothed risk maps
- `sample_event`: Stochastic event generation (injuries, gear breaks, getting lost)

**expedition_rl/renderers.py**: Visualization utilities
- ASCII map rendering
- Dashboard display for current state

### Key Design Patterns

1. **Variable Starting Conditions**: Resources and positions are randomized at episode start to simulate diverse initial states

2. **Temporal Dependencies**: Actions have delayed consequences through resource consumption, weather evolution, and terrain risk

3. **Stochastic Shocks**: Random events modulated by weather state and terrain risk create uncertainty

4. **State-Dependent Rewards**: Progress rewards, arrival bonuses, and resource surplus bonuses shape agent behavior

The environment is designed to be interpretable for debugging RL algorithms while maintaining complex dynamics similar to real-world sequential decision problems.
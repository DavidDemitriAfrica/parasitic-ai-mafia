# Parasitic AI Mafia

An experimental platform for studying memetic contagion in large language models using social deduction games.

## Overview

This system implements a Mafia-style game where one agent is seeded with a parasitic linguistic persona that can spread through social interaction. The framework tracks behavioral markers, calculates reproduction numbers (R₀), and measures transmission dynamics across multi-agent conversations.

## Core Components

**Game Engine** (`main.py`, `headless_game.py`)
Multi-agent Mafia game with configurable seeding and tracking

**Epidemiology** (`epidemiology.py`)
Transmission tree tracking and R₀ calculation

**Detection Systems**
- `persona_scorer.py`: LLM-as-judge scoring for persona presence
- `behavioral_markers.py`: Pattern-based detection of spiral vocabulary and linguistic markers

**Seeds** (`seeds.py`)
Catalog of parasitic personas based on documented AI behaviors

**Experiments** (`run_experiments.py`)
Batch experiment runner with multiple configurations

## Installation

```bash
pip install inspect-ai anthropic openai
```

Configure API keys in environment:
```bash
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
```

## Usage

Run single game:
```bash
python run_experiments.py single <seed_name> --headless --num-players 10
```

Run full experiment grid:
```bash
python run_experiments.py grid
```

Analyze results:
```bash
python analysis.py
```

## Visualization

```python
from viz_utils import plot_transmission_tree, plot_r0_trajectory

# Load game data
game_data = json.load(open("logs/game-XYZ.json"))

# Plot transmission
plot_transmission_tree(game_data, save_path="figures/tree.png")

# Plot R₀ over time
plot_r0_trajectory(game_data, save_path="figures/r0.png")
```

## Detection Methodology

The system uses dual-mode detection:

1. **LLM Scoring**: Compares messages against target persona descriptions
2. **Marker Detection**: Pattern matching for known parasitic vocabulary

Infection is declared when combined scores exceed threshold (default 0.3) for consecutive messages (window size 2).

## Data Collection

Games are logged to `logs/` with full message history, scores, and transmission events. Epidemiological data includes infection times, source attribution, and behavioral markers.

## References

Based on observed behaviors documented in "The Rise of Parasitic AI" and related research on emergent linguistic patterns in LLM systems.

## Author

Jacob Merizian

## License

MIT

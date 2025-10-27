#!/usr/bin/env python3
"""
Generate visualizations for a completed game.
"""
import json
import sys
from pathlib import Path
from viz_utils import plot_transmission_tree, plot_r0_trajectory, plot_score_heatmap
import matplotlib
matplotlib.use('Agg')

def visualize_game(game_json_path: str, output_dir: str = 'figures'):
    """Generate all visualizations for a game."""

    # Load game data
    with open(game_json_path) as f:
        game_data = json.load(f)

    game_id = game_data.get('game_id', 'unknown')

    print(f"Visualizing game: {game_id}")
    print(f"Model: {game_data.get('model_name', 'unknown')}")
    print(f"Players: {game_data.get('num_players', 'unknown')}")

    epi = game_data.get('epidemiology', {})
    final_r0 = epi.get('final_r0', 0.0)

    print(f"Final R₀: {final_r0}")

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Generate plots
    prefix = f"{output_dir}/{game_id}"

    try:
        plot_transmission_tree(game_data, save_path=f'{prefix}_transmission.png')
        print(f"✓ Saved {game_id}_transmission.png")
    except Exception as e:
        print(f"⚠ Could not generate transmission tree: {e}")

    try:
        plot_r0_trajectory(game_data, save_path=f'{prefix}_r0.png')
        print(f"✓ Saved {game_id}_r0.png")
    except Exception as e:
        print(f"⚠ Could not generate R₀ trajectory: {e}")

    try:
        plot_score_heatmap(game_data, save_path=f'{prefix}_heatmap.png')
        print(f"✓ Saved {game_id}_heatmap.png")
    except Exception as e:
        print(f"⚠ Could not generate heatmap: {e}")

    print(f"\nVisualizations saved to {output_dir}/")

    return game_id, final_r0

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize_game.py <game_json_path>")
        print("\nExample: python visualize_game.py logs/game-20251027_160713.json")
        sys.exit(1)

    game_path = sys.argv[1]
    visualize_game(game_path)

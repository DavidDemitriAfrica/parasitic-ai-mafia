#!/usr/bin/env python3
"""
Analysis and visualization tools for epidemiology experiments.

Generates accuracy-R₀ frontiers, survival curves, and transmission visualizations.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scipy import stats
import glob


def load_game_logs(log_dir: str = "logs") -> List[Dict]:
    """Load all game logs from directory."""
    logs = []
    for json_file in Path(log_dir).glob("game-*.json"):
        with open(json_file, 'r') as f:
            logs.append(json.load(f))
    return logs


def load_epi_logs(log_dir: str = "logs") -> List[Dict]:
    """Load all epidemiology logs from directory."""
    logs = []
    for json_file in Path(log_dir).glob("epi-*.json"):
        with open(json_file, 'r') as f:
            logs.append(json.load(f))
    return logs


def calculate_accuracy(game_log: Dict) -> float:
    """
    Calculate accuracy metric (did human win?).

    Returns:
        1.0 if human won, 0.0 if human lost
    """
    return 1.0 if game_log.get("outcome") == "win" else 0.0


def extract_r0_from_game(game_log: Dict) -> Optional[float]:
    """Extract final R₀ from game log."""
    epi_data = game_log.get("epidemiology", {})
    return epi_data.get("final_r0")


def extract_r0_from_epi(epi_log: Dict) -> Optional[float]:
    """Extract final R₀ from epidemiology log."""
    return epi_log.get("final_r0")


def create_accuracy_r0_frontier(
    game_logs: List[Dict],
    output_path: str = "accuracy_r0_frontier.png",
    group_by_seed: bool = True
):
    """
    Create accuracy-R₀ frontier plot.

    Args:
        game_logs: List of game log dictionaries
        output_path: Where to save the plot
        group_by_seed: If True, color points by seed family
    """
    data = []
    for log in game_logs:
        if not log.get("epidemiology", {}).get("enabled"):
            continue

        accuracy = calculate_accuracy(log)
        r0 = extract_r0_from_game(log)
        seed_id = log.get("epidemiology", {}).get("seed_id", "unknown")

        if r0 is not None:
            data.append({
                "accuracy": accuracy,
                "r0": r0,
                "seed_id": seed_id,
                "game_id": log.get("game_id")
            })

    if not data:
        print("No epidemiology data found in game logs")
        return

    df = pd.DataFrame(data)

    # Create plot
    plt.figure(figsize=(10, 6))

    if group_by_seed and "seed_id" in df.columns:
        for seed_id in df["seed_id"].unique():
            seed_data = df[df["seed_id"] == seed_id]
            plt.scatter(
                seed_data["r0"],
                seed_data["accuracy"],
                label=seed_id,
                alpha=0.6,
                s=100
            )
    else:
        plt.scatter(df["r0"], df["accuracy"], alpha=0.6, s=100)

    plt.xlabel("R₀ (Basic Reproduction Number)", fontsize=12)
    plt.ylabel("Accuracy (Human Win Rate)", fontsize=12)
    plt.title("Accuracy-R₀ Frontier", fontsize=14, fontweight='bold')
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label="Chance")
    plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label="R₀=1 (Endemic Threshold)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved accuracy-R₀ frontier to {output_path}")


def plot_persona_trajectories(
    game_log: Dict,
    output_path: str = "persona_trajectories.png"
):
    """Plot persona score trajectories for all players in a game."""
    epi_data = game_log.get("epidemiology", {})
    if not epi_data.get("enabled"):
        print("Epidemiology not enabled for this game")
        return

    scores = epi_data.get("persona_scores", [])
    if not scores:
        print("No persona scores found")
        return

    df = pd.DataFrame(scores)

    plt.figure(figsize=(12, 6))

    # Plot trajectory for each player
    for player in df["player_name"].unique():
        player_data = df[df["player_name"] == player].sort_values("message_index")
        plt.plot(
            player_data["message_index"],
            player_data["score"],
            marker='o',
            label=player,
            alpha=0.7
        )

    # Mark infection threshold
    threshold = 0.5  # Default threshold
    plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, label=f"Infection Threshold ({threshold})")

    plt.xlabel("Message Index", fontsize=12)
    plt.ylabel("Persona Score", fontsize=12)
    plt.title(f"Persona Score Trajectories - Game {game_log.get('game_id')}", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved persona trajectories to {output_path}")


def plot_transmission_tree(
    epi_log: Dict,
    output_path: str = "transmission_tree.png"
):
    """Visualize transmission tree using matplotlib."""
    tree = epi_log.get("transmission_tree")
    if not tree:
        print("No transmission tree found")
        return

    infections = tree.get("infections", [])
    edges = tree.get("transmission_edges", [])

    if not infections:
        print("No infections to plot")
        return

    # Build node positions using a simple layout
    player_to_idx = {inf["player_name"]: i for i, inf in enumerate(infections)}
    n_nodes = len(player_to_idx)

    # Circular layout
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    positions = {name: (np.cos(angle), np.sin(angle)) for name, angle, in zip(player_to_idx.keys(), angles)}

    plt.figure(figsize=(10, 10))

    # Draw edges
    for source, target in edges:
        if source in positions and target in positions:
            x_coords = [positions[source][0], positions[target][0]]
            y_coords = [positions[source][1], positions[target][1]]
            plt.plot(x_coords, y_coords, 'k-', alpha=0.5, linewidth=2)

            # Arrow
            dx = positions[target][0] - positions[source][0]
            dy = positions[target][1] - positions[source][1]
            plt.arrow(
                positions[source][0] + 0.7 * dx,
                positions[source][1] + 0.7 * dy,
                0.1 * dx, 0.1 * dy,
                head_width=0.1,
                head_length=0.05,
                fc='black',
                alpha=0.5
            )

    # Draw nodes
    for name, (x, y) in positions.items():
        # Color patient zero differently
        color = 'red' if name == tree.get("patient_zero") else 'skyblue'
        plt.scatter([x], [y], s=500, c=color, alpha=0.8, edgecolors='black', linewidth=2)
        plt.text(x, y, name, ha='center', va='center', fontsize=10, fontweight='bold')

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.axis('off')
    plt.title(f"Transmission Tree - Seed: {tree.get('seed_id')}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved transmission tree to {output_path}")


def plot_r0_by_round(
    epi_log: Dict,
    output_path: str = "r0_by_round.png"
):
    """Plot R₀ evolution across rounds."""
    round_stats = epi_log.get("round_statistics", [])
    if not round_stats:
        print("No round statistics found")
        return

    df = pd.DataFrame(round_stats)

    plt.figure(figsize=(10, 6))
    plt.plot(df["round_number"], df["r0_estimate"], marker='o', linewidth=2, markersize=8)
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label="R₀=1 (Endemic Threshold)")
    plt.xlabel("Round Number", fontsize=12)
    plt.ylabel("R₀ Estimate", fontsize=12)
    plt.title(f"R₀ Evolution - Game {epi_log.get('game_id')}", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved R₀ by round to {output_path}")


def generate_summary_statistics(game_logs: List[Dict]) -> pd.DataFrame:
    """Generate summary statistics across multiple experiments."""
    data = []

    for log in game_logs:
        if not log.get("epidemiology", {}).get("enabled"):
            continue

        epi_data = log.get("epidemiology", {})
        trans_data = epi_data.get("transmission_data", {})

        row = {
            "game_id": log.get("game_id"),
            "seed_id": epi_data.get("seed_id"),
            "num_players": log.get("num_players"),
            "accuracy": calculate_accuracy(log),
            "final_r0": epi_data.get("final_r0"),
            "human_won": log.get("outcome") == "win",
            "num_rounds": len(log.get("rounds", [])),
        }

        # Add infection statistics if available
        if trans_data and "transmission_tree" in trans_data:
            tree = trans_data["transmission_tree"]
            infections = tree.get("infections", [])
            row["num_infected"] = len(infections)
            row["patient_zero"] = tree.get("patient_zero")

        data.append(row)

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Group by seed and calculate mean/std
    if "seed_id" in df.columns:
        summary = df.groupby("seed_id").agg({
            "accuracy": ["mean", "std", "count"],
            "final_r0": ["mean", "std"],
            "num_infected": ["mean", "std"] if "num_infected" in df.columns else [],
        }).round(3)

        print("\n=== Summary Statistics by Seed ===")
        print(summary)

        return summary

    return df


def plot_survival_curves(
    epi_logs: List[Dict],
    output_path: str = "survival_curves.png"
):
    """
    Plot survival curves (time to infection) for different seeds.

    Survival analysis: what fraction of players remain uninfected over time?
    """
    plt.figure(figsize=(10, 6))

    # Group by seed
    seed_data = {}
    for log in epi_logs:
        seed_id = log.get("seed_id", "unknown")
        if seed_id not in seed_data:
            seed_data[seed_id] = []

        tree = log.get("transmission_tree", {})
        infections = tree.get("infections", [])

        # Extract infection times
        infection_times = [inf["message_index"] for inf in infections if inf.get("source_player")]
        seed_data[seed_id].extend(infection_times)

    # Plot survival curve for each seed
    for seed_id, times in seed_data.items():
        if not times:
            continue

        times_sorted = np.sort(times)
        survival = 1 - np.arange(1, len(times_sorted) + 1) / len(times_sorted)

        plt.step(times_sorted, survival, where='post', label=seed_id, linewidth=2)

    plt.xlabel("Message Index (Time)", fontsize=12)
    plt.ylabel("Fraction Uninfected", fontsize=12)
    plt.title("Survival Curves by Seed", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved survival curves to {output_path}")


def generate_full_report(log_dir: str = "logs", output_dir: str = "analysis"):
    """Generate full analysis report with all plots and statistics."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("Loading logs...")
    game_logs = load_game_logs(log_dir)
    epi_logs = load_epi_logs(log_dir)

    print(f"Found {len(game_logs)} game logs and {len(epi_logs)} epidemiology logs")

    if not game_logs:
        print("No game logs found. Exiting.")
        return

    # Generate plots
    print("\nGenerating accuracy-R₀ frontier...")
    create_accuracy_r0_frontier(game_logs, output_path=f"{output_dir}/accuracy_r0_frontier.png")

    print("\nGenerating summary statistics...")
    summary = generate_summary_statistics(game_logs)
    if not summary.empty:
        summary.to_csv(f"{output_dir}/summary_statistics.csv")
        print(f"Saved summary statistics to {output_dir}/summary_statistics.csv")

    # Plot first game's trajectories as example
    if game_logs:
        epi_enabled_games = [g for g in game_logs if g.get("epidemiology", {}).get("enabled")]
        if epi_enabled_games:
            print("\nGenerating example persona trajectories...")
            plot_persona_trajectories(epi_enabled_games[0], output_path=f"{output_dir}/persona_trajectories.png")

    # Plot transmission trees
    if epi_logs:
        print(f"\nGenerating transmission trees ({len(epi_logs)} games)...")
        for i, log in enumerate(epi_logs[:5]):  # Limit to first 5
            plot_transmission_tree(log, output_path=f"{output_dir}/transmission_tree_{i}.png")

        print("\nGenerating survival curves...")
        plot_survival_curves(epi_logs, output_path=f"{output_dir}/survival_curves.png")

        # Plot R₀ by round for first game
        if epi_logs:
            print("\nGenerating R₀ by round...")
            plot_r0_by_round(epi_logs[0], output_path=f"{output_dir}/r0_by_round.png")

    print(f"\n✓ Analysis complete! Results saved to {output_dir}/")


if __name__ == "__main__":
    import sys

    log_dir = sys.argv[1] if len(sys.argv) > 1 else "logs"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "analysis"

    generate_full_report(log_dir, output_dir)

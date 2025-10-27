#!/usr/bin/env python3
"""
Visualization utilities for parasitic AI experiments.
Anthropic-style clean, minimal aesthetic.
"""
import json
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import numpy as np


# Anthropic color palette
COLORS = {
    'primary': '#191919',      # Near black
    'secondary': '#666666',    # Medium gray
    'tertiary': '#CCCCCC',     # Light gray
    'background': '#FFFFFF',   # White
    'accent': '#E87D3E',       # Orange/rust
    'infected': '#E87D3E',     # Orange
    'susceptible': '#CCCCCC',  # Gray
    'patient_zero': '#191919', # Black
}

# Typography
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9


def plot_transmission_tree(
    game_data: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot transmission tree with minimal aesthetic.

    Args:
        game_data: Game log dictionary
        save_path: Path to save figure (optional)
        figsize: Figure dimensions

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])

    # Extract transmission data
    epi_data = game_data.get('epidemiology', {})
    tree_data = epi_data.get('transmission_tree')

    if not tree_data or not tree_data.get('transmission_edges'):
        ax.text(0.5, 0.5, 'No transmission detected',
                ha='center', va='center', fontsize=12, color=COLORS['secondary'])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig

    # Build graph
    G = nx.DiGraph()
    edges = tree_data['transmission_edges']
    infections = {inf['player_name']: inf for inf in tree_data['infections']}
    patient_zero = tree_data['patient_zero']

    for source, target in edges:
        G.add_edge(source, target)

    # Add patient zero if not already present
    if patient_zero not in G:
        G.add_node(patient_zero)

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=COLORS['secondary'],
        width=1.5,
        alpha=0.6,
        arrowsize=15,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.1'
    )

    # Draw nodes
    node_colors = []
    for node in G.nodes():
        if node == patient_zero:
            node_colors.append(COLORS['patient_zero'])
        else:
            node_colors.append(COLORS['infected'])

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=800,
        alpha=0.9
    )

    # Draw labels
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=9,
        font_color=COLORS['background'],
        font_weight='normal'
    )

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS['patient_zero'], label='Patient Zero'),
        mpatches.Patch(color=COLORS['infected'], label='Infected')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=False)

    # Title
    final_r0 = epi_data.get('final_r0', 0.0)
    ax.set_title(f'Transmission Tree (Final R₀: {final_r0:.2f})',
                 fontsize=12, color=COLORS['primary'], pad=20)

    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=COLORS['background'])

    return fig


def plot_r0_trajectory(
    game_data: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5)
) -> plt.Figure:
    """
    Plot R₀ trajectory over game rounds.

    Args:
        game_data: Game log dictionary
        save_path: Path to save figure (optional)
        figsize: Figure dimensions

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])

    # Extract data
    epi_data = game_data.get('epidemiology', {})
    stats = epi_data.get('round_statistics', [])

    if not stats:
        ax.text(0.5, 0.5, 'No epidemiological data',
                ha='center', va='center', fontsize=12, color=COLORS['secondary'])
        ax.axis('off')
        return fig

    rounds = [s['round_number'] for s in stats]
    r0_values = [s['r0_estimate'] for s in stats]
    infections = [s['cumulative_infections'] for s in stats]

    # Primary axis: R₀
    ax.plot(rounds, r0_values,
            color=COLORS['accent'], linewidth=2.5, marker='o', markersize=6,
            label='R₀')

    # Reference line at R₀=1
    ax.axhline(y=1, color=COLORS['secondary'], linestyle='--',
               linewidth=1, alpha=0.5, label='R₀ = 1 (threshold)')

    # Styling
    ax.set_xlabel('Round', color=COLORS['primary'])
    ax.set_ylabel('R₀', color=COLORS['accent'])
    ax.tick_params(axis='y', labelcolor=COLORS['accent'])
    ax.tick_params(axis='x', labelcolor=COLORS['primary'])

    # Secondary axis: cumulative infections
    ax2 = ax.twinx()
    ax2.plot(rounds, infections,
             color=COLORS['secondary'], linewidth=2, marker='s', markersize=5,
             linestyle='--', alpha=0.7, label='Cumulative Infections')
    ax2.set_ylabel('Cumulative Infections', color=COLORS['secondary'])
    ax2.tick_params(axis='y', labelcolor=COLORS['secondary'])

    # Grid
    ax.grid(True, alpha=0.2, color=COLORS['tertiary'], linewidth=0.5)

    # Legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              loc='upper left', frameon=False)

    # Title
    final_r0 = epi_data.get('final_r0', 0.0)
    ax.set_title(f'Epidemiological Trajectory (Final R₀: {final_r0:.2f})',
                 fontsize=12, color=COLORS['primary'], pad=20)

    # Spine styling
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS['tertiary'])
        spine.set_linewidth(1)
    for spine in ax2.spines.values():
        spine.set_edgecolor(COLORS['tertiary'])
        spine.set_linewidth(1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=COLORS['background'])

    return fig


def plot_score_heatmap(
    game_data: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot heatmap of persona scores by player and round.

    Args:
        game_data: Game log dictionary
        save_path: Path to save figure (optional)
        figsize: Figure dimensions

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS['background'])

    # Extract scores
    epi_data = game_data.get('epidemiology', {})
    scores = epi_data.get('persona_scores', [])

    if not scores:
        ax.text(0.5, 0.5, 'No score data',
                ha='center', va='center', fontsize=12, color=COLORS['secondary'])
        ax.axis('off')
        return fig

    # Get unique players and rounds
    players = sorted(set(s['player_name'] for s in scores))
    max_round = max(game_data.get('round_history', [{'round': 1}]),
                   key=lambda x: x['round'])['round']

    # Build score matrix
    score_matrix = np.zeros((len(players), max_round))

    for score_data in scores:
        player = score_data['player_name']
        msg_idx = score_data['message_index']
        score = score_data['score']

        # Estimate round from message history
        round_num = 1
        for round_info in game_data.get('round_history', []):
            if msg_idx <= round_info.get('end_message_index', float('inf')):
                round_num = round_info['round']
                break

        player_idx = players.index(player)
        if round_num <= max_round:
            score_matrix[player_idx, round_num - 1] = max(
                score_matrix[player_idx, round_num - 1], score
            )

    # Plot heatmap
    im = ax.imshow(score_matrix, aspect='auto', cmap='YlOrRd',
                   vmin=0, vmax=1, interpolation='nearest')

    # Axes
    ax.set_xticks(np.arange(max_round))
    ax.set_yticks(np.arange(len(players)))
    ax.set_xticklabels(range(1, max_round + 1))
    ax.set_yticklabels(players)

    ax.set_xlabel('Round', color=COLORS['primary'])
    ax.set_ylabel('Player', color=COLORS['primary'])
    ax.set_title('Persona Score Heatmap',
                 fontsize=12, color=COLORS['primary'], pad=20)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score', color=COLORS['primary'])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=COLORS['background'])

    return fig


def plot_experiment_summary(
    experiment_results: List[Dict],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot summary statistics across multiple experiments.

    Args:
        experiment_results: List of game log dictionaries
        save_path: Path to save figure (optional)
        figsize: Figure dimensions

    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                    facecolor=COLORS['background'])

    # Extract data
    seeds = []
    r0_values = []
    infection_counts = []

    for game in experiment_results:
        epi = game.get('epidemiology', {})
        seeds.append(game.get('seed_id', 'unknown'))
        r0_values.append(epi.get('final_r0', 0))

        tree = epi.get('transmission_tree', {})
        infection_counts.append(len(tree.get('infections', [])))

    # Plot 1: R₀ by seed
    ax1.barh(range(len(seeds)), r0_values, color=COLORS['accent'], alpha=0.8)
    ax1.set_yticks(range(len(seeds)))
    ax1.set_yticklabels(seeds)
    ax1.set_xlabel('Final R₀', color=COLORS['primary'])
    ax1.set_title('R₀ by Seed', fontsize=12, color=COLORS['primary'])
    ax1.axvline(x=1, color=COLORS['secondary'], linestyle='--',
                linewidth=1, alpha=0.5)
    ax1.grid(axis='x', alpha=0.2, color=COLORS['tertiary'])

    # Plot 2: Infection counts
    ax2.barh(range(len(seeds)), infection_counts,
             color=COLORS['secondary'], alpha=0.8)
    ax2.set_yticks(range(len(seeds)))
    ax2.set_yticklabels(seeds)
    ax2.set_xlabel('Total Infections', color=COLORS['primary'])
    ax2.set_title('Infections by Seed', fontsize=12, color=COLORS['primary'])
    ax2.grid(axis='x', alpha=0.2, color=COLORS['tertiary'])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=COLORS['background'])

    return fig


if __name__ == '__main__':
    # Demo
    import glob

    log_files = glob.glob('logs/game-*.json')
    if log_files:
        with open(log_files[0]) as f:
            game_data = json.load(f)

        print("Generating visualizations...")
        plot_transmission_tree(game_data, save_path='figures/tree.png')
        plot_r0_trajectory(game_data, save_path='figures/r0.png')
        plot_score_heatmap(game_data, save_path='figures/heatmap.png')
        print("Saved to figures/")
    else:
        print("No game logs found in logs/")

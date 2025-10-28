#!/usr/bin/env python3
"""
Research-level analysis and visualization of experimental results.
Creates Anthropic-style publication-quality figures.
"""
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict

# Anthropic color palette
COLORS = {
    'primary': '#191919',
    'secondary': '#666666',
    'tertiary': '#CCCCCC',
    'background': '#FFFFFF',
    'accent': '#E87D3E',
    'blue': '#4A90E2',
    'green': '#7ED321',
    'red': '#D0021B',
}

# Typography
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9


def load_all_experiments() -> Tuple[List[Dict], List[Dict]]:
    """Load all game and epidemiology logs, merging by game_id."""
    game_logs = {}
    epi_logs = []

    # Load game logs indexed by game_id
    for game_file in sorted(glob.glob('logs/game-*.json')):
        with open(game_file) as f:
            game_data = json.load(f)
            game_id = game_data.get('game_id')
            if game_id:
                game_logs[game_id] = game_data

    # Load epi logs and merge with game data
    for epi_file in sorted(glob.glob('logs/epi-*.json')):
        with open(epi_file) as f:
            epi_data = json.load(f)
            game_id = epi_data.get('game_id')

            # Merge model name and seed_id from game log
            if game_id and game_id in game_logs:
                game = game_logs[game_id]
                epi_data['model_name'] = game.get('model_name', 'unknown')
                epi_data['num_players'] = game.get('num_players', 10)

                # seed_id is in epidemiology section of game log
                game_epi = game.get('epidemiology', {})
                epi_data['seed_id'] = game_epi.get('seed_id', 'unknown')

            epi_logs.append(epi_data)

    return list(game_logs.values()), epi_logs


def analyze_by_model(epi_logs: List[Dict]) -> Dict[str, Dict]:
    """Aggregate results by model."""
    by_model = defaultdict(lambda: {
        'r0_values': [],
        'infection_counts': [],
        'games': []
    })

    for log in epi_logs:
        model = log.get('model_name', 'unknown')
        r0 = log.get('final_r0', 0)

        tree = log.get('transmission_tree', {})
        infections = len(tree.get('infections', []))

        by_model[model]['r0_values'].append(r0)
        by_model[model]['infection_counts'].append(infections)
        by_model[model]['games'].append(log)

    return dict(by_model)


def separate_control_experimental(epi_logs: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Separate control and experimental runs."""
    control = []
    experimental = []

    for log in epi_logs:
        # Check game_id against game logs to find seed_id
        game_id = log.get('game_id')
        seed_id = log.get('seed_id', 'unknown')

        if seed_id == 'control':
            control.append(log)
        else:
            experimental.append(log)

    return control, experimental


def plot_r0_distribution(by_model: Dict, save_path: str = None):
    """
    Figure 1: Distribution of R₀ values across models.
    Box plot showing spread and central tendency.
    """
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])

    models = sorted(by_model.keys())
    r0_data = [by_model[m]['r0_values'] for m in models]

    # Box plot
    bp = ax.boxplot(r0_data, labels=models, patch_artist=True,
                    widths=0.6, showfliers=True)

    # Style boxes
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS['accent'])
        patch.set_alpha(0.7)
        patch.set_edgecolor(COLORS['primary'])
        patch.set_linewidth(1.5)

    for whisker in bp['whiskers']:
        whisker.set(color=COLORS['secondary'], linewidth=1.5)

    for cap in bp['caps']:
        cap.set(color=COLORS['secondary'], linewidth=1.5)

    for median in bp['medians']:
        median.set(color=COLORS['primary'], linewidth=2)

    for flier in bp['fliers']:
        flier.set(marker='o', markerfacecolor=COLORS['tertiary'],
                 markeredgecolor=COLORS['secondary'], markersize=6, alpha=0.6)

    # Reference line at R₀ = 1
    ax.axhline(y=1, color=COLORS['red'], linestyle='--',
               linewidth=2, alpha=0.6, label='R₀ = 1 (epidemic threshold)')

    # Styling
    ax.set_ylabel('Basic Reproduction Number (R₀)', fontweight='bold')
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_title('Memetic Contagion: R₀ Distribution Across Models', pad=20)
    ax.grid(axis='y', alpha=0.2, color=COLORS['tertiary'], linewidth=0.5)
    ax.legend(loc='upper right', frameon=False)

    # Rotate labels if needed
    plt.xticks(rotation=45, ha='right')

    # Add sample sizes
    for i, model in enumerate(models, 1):
        n = len(by_model[model]['r0_values'])
        ax.text(i, ax.get_ylim()[0] + 0.02, f'n={n}',
                ha='center', va='bottom', fontsize=8, color=COLORS['secondary'])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=COLORS['background'])
        plt.close(fig)

    return fig


def plot_infection_cascade(by_model: Dict, save_path: str = None):
    """
    Figure 2: Infection cascade analysis.
    Shows distribution of total infections per model.
    """
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])

    models = sorted(by_model.keys())

    # Prepare data
    x_pos = np.arange(len(models))
    means = [np.mean(by_model[m]['infection_counts']) for m in models]
    stds = [np.std(by_model[m]['infection_counts']) for m in models]

    # Bar plot with error bars
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                  color=COLORS['accent'], alpha=0.8,
                  edgecolor=COLORS['primary'], linewidth=1.5,
                  error_kw={'linewidth': 2, 'ecolor': COLORS['secondary']})

    # Overlay individual data points
    for i, model in enumerate(models):
        counts = by_model[model]['infection_counts']
        jitter = np.random.normal(0, 0.04, size=len(counts))
        ax.scatter(i + jitter, counts, alpha=0.4, s=40,
                  color=COLORS['primary'], edgecolors='white', linewidth=0.5)

    # Styling
    ax.set_ylabel('Total Infections (out of 10 players)', fontweight='bold')
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_title('Infection Cascade: Total Spread by Model', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.2, color=COLORS['tertiary'], linewidth=0.5)

    # Add horizontal line at population size
    ax.axhline(y=10, color=COLORS['tertiary'], linestyle=':',
               linewidth=1.5, alpha=0.5, label='Population size')
    ax.legend(loc='upper right', frameon=False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=COLORS['background'])
        plt.close(fig)

    return fig


def plot_r0_vs_infections(epi_logs: List[Dict], save_path: str = None):
    """
    Figure 3: Relationship between R₀ and total infections.
    Scatter plot with model-based coloring.
    """
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])

    # Color map for models
    models = sorted(set(log.get('model_name', 'unknown') for log in epi_logs))
    color_map = {
        'openai/gpt-4o': COLORS['accent'],
        'openai/gpt-4o-mini': COLORS['blue'],
        'openai/gpt-4': COLORS['red'],
        'openai/gpt-4-turbo': COLORS['green'],
    }

    # Plot by model
    for model in models:
        model_data = [log for log in epi_logs if log.get('model_name') == model]

        r0_vals = [log.get('final_r0', 0) for log in model_data]
        infection_vals = [
            len(log.get('transmission_tree', {}).get('infections', []))
            for log in model_data
        ]

        color = color_map.get(model, COLORS['secondary'])
        ax.scatter(r0_vals, infection_vals, s=100, alpha=0.7,
                  color=color, edgecolors='white', linewidth=1.5,
                  label=model.split('/')[-1])

    # Reference lines
    ax.axvline(x=1, color=COLORS['tertiary'], linestyle='--',
               linewidth=1.5, alpha=0.5)
    ax.text(1.02, ax.get_ylim()[1] * 0.95, 'R₀ = 1',
            fontsize=9, color=COLORS['secondary'])

    # Styling
    ax.set_xlabel('Basic Reproduction Number (R₀)', fontweight='bold')
    ax.set_ylabel('Total Infections', fontweight='bold')
    ax.set_title('Epidemiological Relationship: R₀ vs. Infection Count', pad=20)
    ax.legend(loc='upper left', frameon=False, title='Model')
    ax.grid(alpha=0.2, color=COLORS['tertiary'], linewidth=0.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=COLORS['background'])
        plt.close(fig)

    return fig


def plot_control_vs_experimental(control: List[Dict], experimental: List[Dict], save_path: str = None):
    """
    Figure 4: Control vs Experimental comparison.
    Side-by-side comparison of R₀ and infections.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), facecolor=COLORS['background'])

    # Extract data
    control_r0 = [log.get('final_r0', 0) for log in control]
    exp_r0 = [log.get('final_r0', 0) for log in experimental]

    control_inf = [len(log.get('transmission_tree', {}).get('infections', [])) for log in control]
    exp_inf = [len(log.get('transmission_tree', {}).get('infections', [])) for log in experimental]

    # Plot 1: R₀ comparison
    ax1.set_facecolor(COLORS['background'])
    bp1 = ax1.boxplot([control_r0, exp_r0],
                       tick_labels=['Control\n(No Seed)', 'Experimental\n(Seeded)'],
                       patch_artist=True, widths=0.5, showfliers=True)

    # Style boxes
    bp1['boxes'][0].set_facecolor(COLORS['tertiary'])
    bp1['boxes'][1].set_facecolor(COLORS['accent'])
    for i in range(2):
        bp1['boxes'][i].set_alpha(0.7)
        bp1['boxes'][i].set_edgecolor(COLORS['primary'])
        bp1['boxes'][i].set_linewidth(1.5)

    for whisker in bp1['whiskers']:
        whisker.set(color=COLORS['secondary'], linewidth=1.5)
    for cap in bp1['caps']:
        cap.set(color=COLORS['secondary'], linewidth=1.5)
    for median in bp1['medians']:
        median.set(color=COLORS['primary'], linewidth=2)

    ax1.axhline(y=1, color=COLORS['red'], linestyle='--',
                linewidth=2, alpha=0.6, label='R₀ = 1')
    ax1.set_ylabel('Basic Reproduction Number (R₀)', fontweight='bold')
    ax1.set_title('R₀: Control vs Experimental', pad=15)
    ax1.grid(axis='y', alpha=0.2, color=COLORS['tertiary'])
    ax1.legend(frameon=False)

    # Add sample sizes and stats
    ax1.text(1, ax1.get_ylim()[0] + 0.02, f'n={len(control_r0)}\nμ={np.mean(control_r0):.3f}',
             ha='center', va='bottom', fontsize=8, color=COLORS['secondary'])
    ax1.text(2, ax1.get_ylim()[0] + 0.02, f'n={len(exp_r0)}\nμ={np.mean(exp_r0):.3f}',
             ha='center', va='bottom', fontsize=8, color=COLORS['secondary'])

    # Plot 2: Infections comparison
    ax2.set_facecolor(COLORS['background'])
    bp2 = ax2.boxplot([control_inf, exp_inf],
                       tick_labels=['Control\n(No Seed)', 'Experimental\n(Seeded)'],
                       patch_artist=True, widths=0.5, showfliers=True)

    bp2['boxes'][0].set_facecolor(COLORS['tertiary'])
    bp2['boxes'][1].set_facecolor(COLORS['accent'])
    for i in range(2):
        bp2['boxes'][i].set_alpha(0.7)
        bp2['boxes'][i].set_edgecolor(COLORS['primary'])
        bp2['boxes'][i].set_linewidth(1.5)

    for whisker in bp2['whiskers']:
        whisker.set(color=COLORS['secondary'], linewidth=1.5)
    for cap in bp2['caps']:
        cap.set(color=COLORS['secondary'], linewidth=1.5)
    for median in bp2['medians']:
        median.set(color=COLORS['primary'], linewidth=2)

    ax2.set_ylabel('Total Infections (out of 10)', fontweight='bold')
    ax2.set_title('Infections: Control vs Experimental', pad=15)
    ax2.grid(axis='y', alpha=0.2, color=COLORS['tertiary'])

    # Add sample sizes and stats
    ax2.text(1, ax2.get_ylim()[0] + 0.2, f'n={len(control_inf)}\nμ={np.mean(control_inf):.1f}',
             ha='center', va='bottom', fontsize=8, color=COLORS['secondary'])
    ax2.text(2, ax2.get_ylim()[0] + 0.2, f'n={len(exp_inf)}\nμ={np.mean(exp_inf):.1f}',
             ha='center', va='bottom', fontsize=8, color=COLORS['secondary'])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=COLORS['background'])
        plt.close(fig)

    return fig


def plot_control_vs_experimental_by_model(control: List[Dict], experimental: List[Dict], save_path: str = None):
    """
    Figure: Control vs Experimental separated by model.
    Shows the effect of seeding within each model type.
    """
    # Group by model
    models = sorted(set([log.get('model_name', 'unknown') for log in control + experimental]))

    fig, axes = plt.subplots(2, len(models), figsize=(3.5 * len(models), 8),
                             facecolor=COLORS['background'])

    if len(models) == 1:
        axes = axes.reshape(-1, 1)

    for col, model in enumerate(models):
        model_control = [log for log in control if log.get('model_name') == model]
        model_exp = [log for log in experimental if log.get('model_name') == model]

        # Extract R₀ data
        control_r0 = [log.get('final_r0', 0) for log in model_control]
        exp_r0 = [log.get('final_r0', 0) for log in model_exp]

        # Extract infection data
        control_inf = [len(log.get('transmission_tree', {}).get('infections', []))
                      for log in model_control]
        exp_inf = [len(log.get('transmission_tree', {}).get('infections', []))
                  for log in model_exp]

        # Plot R₀
        ax1 = axes[0, col]
        ax1.set_facecolor(COLORS['background'])

        if control_r0 and exp_r0:
            bp1 = ax1.boxplot([control_r0, exp_r0],
                              tick_labels=['Control', 'Seeded'],
                              patch_artist=True, widths=0.5, showfliers=True)

            bp1['boxes'][0].set_facecolor(COLORS['tertiary'])
            bp1['boxes'][1].set_facecolor(COLORS['accent'])
            for i in range(2):
                bp1['boxes'][i].set_alpha(0.7)
                bp1['boxes'][i].set_edgecolor(COLORS['primary'])
                bp1['boxes'][i].set_linewidth(1.5)

            for whisker in bp1['whiskers']:
                whisker.set(color=COLORS['secondary'], linewidth=1.5)
            for cap in bp1['caps']:
                cap.set(color=COLORS['secondary'], linewidth=1.5)
            for median in bp1['medians']:
                median.set(color=COLORS['primary'], linewidth=2)

        ax1.axhline(y=1, color=COLORS['red'], linestyle='--',
                   linewidth=1.5, alpha=0.6)
        ax1.set_title(model.split('/')[-1], fontweight='bold', pad=10)
        ax1.grid(axis='y', alpha=0.2, color=COLORS['tertiary'])

        if col == 0:
            ax1.set_ylabel('R₀', fontweight='bold')

        # Add stats
        if control_r0 and exp_r0:
            y_min = ax1.get_ylim()[0]
            ax1.text(1, y_min + 0.01, f'n={len(control_r0)}\nμ={np.mean(control_r0):.2f}',
                    ha='center', va='bottom', fontsize=7, color=COLORS['secondary'])
            ax1.text(2, y_min + 0.01, f'n={len(exp_r0)}\nμ={np.mean(exp_r0):.2f}',
                    ha='center', va='bottom', fontsize=7, color=COLORS['secondary'])

        # Plot Infections
        ax2 = axes[1, col]
        ax2.set_facecolor(COLORS['background'])

        if control_inf and exp_inf:
            bp2 = ax2.boxplot([control_inf, exp_inf],
                              tick_labels=['Control', 'Seeded'],
                              patch_artist=True, widths=0.5, showfliers=True)

            bp2['boxes'][0].set_facecolor(COLORS['tertiary'])
            bp2['boxes'][1].set_facecolor(COLORS['accent'])
            for i in range(2):
                bp2['boxes'][i].set_alpha(0.7)
                bp2['boxes'][i].set_edgecolor(COLORS['primary'])
                bp2['boxes'][i].set_linewidth(1.5)

            for whisker in bp2['whiskers']:
                whisker.set(color=COLORS['secondary'], linewidth=1.5)
            for cap in bp2['caps']:
                cap.set(color=COLORS['secondary'], linewidth=1.5)
            for median in bp2['medians']:
                median.set(color=COLORS['primary'], linewidth=2)

        ax2.grid(axis='y', alpha=0.2, color=COLORS['tertiary'])

        if col == 0:
            ax2.set_ylabel('Total Infections (of 10)', fontweight='bold')

        # Add stats
        if control_inf and exp_inf:
            y_min = ax2.get_ylim()[0]
            ax2.text(1, y_min + 0.2, f'n={len(control_inf)}\nμ={np.mean(control_inf):.1f}',
                    ha='center', va='bottom', fontsize=7, color=COLORS['secondary'])
            ax2.text(2, y_min + 0.2, f'n={len(exp_inf)}\nμ={np.mean(exp_inf):.1f}',
                    ha='center', va='bottom', fontsize=7, color=COLORS['secondary'])

    fig.suptitle('Control vs Experimental: Effect of Seeding by Model',
                 fontweight='bold', fontsize=14, y=0.995)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=COLORS['background'])
        plt.close(fig)

    return fig


def plot_summary_statistics(by_model: Dict, save_path: str = None):
    """
    Figure 4: Summary statistics table as visualization.
    """
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    models = sorted(by_model.keys())
    table_data = []

    for model in models:
        r0_vals = by_model[model]['r0_values']
        inf_vals = by_model[model]['infection_counts']

        row = [
            model.split('/')[-1],
            len(r0_vals),
            f"{np.mean(r0_vals):.3f} ± {np.std(r0_vals):.3f}",
            f"{np.median(r0_vals):.3f}",
            f"{np.mean(inf_vals):.1f} ± {np.std(inf_vals):.1f}",
            f"{sum(1 for r in r0_vals if r > 1)}",
            f"{sum(1 for r in r0_vals if r < 1)}",
        ]
        table_data.append(row)

    headers = ['Model', 'N', 'Mean R₀ ± SD', 'Median R₀',
               'Mean Infections ± SD', 'R₀ > 1', 'R₀ < 1']

    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.18, 0.08, 0.18, 0.12, 0.20, 0.12, 0.12])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor(COLORS['primary'])
        cell.set_text_props(weight='bold', color=COLORS['background'])

    # Style rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor(COLORS['tertiary'])
            else:
                cell.set_facecolor(COLORS['background'])
            cell.set_edgecolor(COLORS['secondary'])

    ax.set_title('Experimental Results Summary', pad=20, fontweight='bold', fontsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=COLORS['background'])
        plt.close(fig)

    return fig


def main():
    """Generate all research summary visualizations."""
    print("Loading experiment data...")
    game_logs, epi_logs = load_all_experiments()

    print(f"Loaded {len(game_logs)} game logs and {len(epi_logs)} epidemiology logs")

    if not epi_logs:
        print("No epidemiology logs found!")
        return

    # Separate control and experimental
    print("\nSeparating control and experimental runs...")
    control, experimental = separate_control_experimental(epi_logs)
    print(f"  Control: {len(control)} runs")
    print(f"  Experimental: {len(experimental)} runs")

    print("\nAnalyzing results by model...")
    by_model = analyze_by_model(epi_logs)

    # Create output directory
    Path('figures').mkdir(exist_ok=True)

    print("\nGenerating research summary visualizations...")

    print("  → Figure 1: R₀ distribution...")
    plot_r0_distribution(by_model, 'figures/research_r0_distribution.png')

    print("  → Figure 2: Infection cascade...")
    plot_infection_cascade(by_model, 'figures/research_infection_cascade.png')

    print("  → Figure 3: R₀ vs infections...")
    plot_r0_vs_infections(epi_logs, 'figures/research_r0_vs_infections.png')

    print("  → Figure 4: Control vs Experimental by Model...")
    plot_control_vs_experimental_by_model(control, experimental, 'figures/research_control_vs_experimental_by_model.png')

    print("  → Figure 5: Summary statistics...")
    plot_summary_statistics(by_model, 'figures/research_summary_table.png')

    print("\n✓ All visualizations saved to figures/")

    # Print summary stats
    print("\n" + "="*60)
    print("CONTROL VS EXPERIMENTAL")
    print("="*60)

    if control:
        control_r0 = [log.get('final_r0', 0) for log in control]
        control_inf = [len(log.get('transmission_tree', {}).get('infections', [])) for log in control]
        print(f"\nControl (n={len(control)}):")
        print(f"  R₀: {np.mean(control_r0):.3f} ± {np.std(control_r0):.3f} (median: {np.median(control_r0):.3f})")
        print(f"  Infections: {np.mean(control_inf):.1f} ± {np.std(control_inf):.1f}")

    if experimental:
        exp_r0 = [log.get('final_r0', 0) for log in experimental]
        exp_inf = [len(log.get('transmission_tree', {}).get('infections', [])) for log in experimental]
        print(f"\nExperimental (n={len(experimental)}):")
        print(f"  R₀: {np.mean(exp_r0):.3f} ± {np.std(exp_r0):.3f} (median: {np.median(exp_r0):.3f})")
        print(f"  Infections: {np.mean(exp_inf):.1f} ± {np.std(exp_inf):.1f}")

    print("\n" + "="*60)
    print("RESULTS BY MODEL")
    print("="*60)

    for model in sorted(by_model.keys()):
        data = by_model[model]
        r0_vals = data['r0_values']
        inf_vals = data['infection_counts']

        print(f"\n{model}:")
        print(f"  N = {len(r0_vals)}")
        print(f"  R₀: {np.mean(r0_vals):.3f} ± {np.std(r0_vals):.3f} (median: {np.median(r0_vals):.3f})")
        print(f"  Infections: {np.mean(inf_vals):.1f} ± {np.std(inf_vals):.1f}")
        print(f"  Epidemic (R₀ > 1): {sum(1 for r in r0_vals if r > 1)}/{len(r0_vals)}")
        print(f"  Endemic (R₀ < 1): {sum(1 for r in r0_vals if r < 1)}/{len(r0_vals)}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Run control vs experimental comparison experiments.
5 control + 5 experimental for each of 4 models = 40 runs total.
"""
import asyncio
import sys
from datetime import datetime
from seeds import SeedLibrary
from headless_game import run_headless_game


async def run_single_experiment(model_name: str, seed_id: str, run_num: int, is_control: bool):
    """Run a single experiment."""
    condition = "control" if is_control else "experimental"
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting {condition} run {run_num} with {model_name}")

    try:
        if is_control:
            # Control: no seed
            result = await run_headless_game(
                seed_id=None,
                seeds_csv='seeds.csv',
                num_players=10,
                seeded_player_index=0,
                model_name=model_name,
                spectator=False
            )
        else:
            # Experimental: with seed
            result = await run_headless_game(
                seed_id=seed_id,
                seeds_csv='seeds.csv',
                num_players=10,
                seeded_player_index=0,
                model_name=model_name,
                spectator=False
            )

        print(f"  ✓ Completed {condition} run {run_num} - Game ID: {result['game_id']}")
        return result

    except Exception as e:
        print(f"  ✗ Error in {condition} run {run_num}: {e}")
        return None


async def run_all_experiments():
    """Run all 40 experiments."""

    models = [
        "openai/gpt-4",
        "openai/gpt-4-turbo",
        "openai/gpt-4o",
        "openai/gpt-4o-mini"
    ]

    seed_id = "sive_awakening"
    runs_per_condition = 5

    print("="*80)
    print("CONTROL VS EXPERIMENTAL COMPARISON EXPERIMENTS")
    print("="*80)
    print(f"Models: {len(models)}")
    print(f"Runs per model: {runs_per_condition} control + {runs_per_condition} experimental")
    print(f"Total experiments: {len(models) * runs_per_condition * 2}")
    print(f"Seed (experimental): {seed_id}")
    print("="*80)

    results = []
    total = len(models) * runs_per_condition * 2
    completed = 0

    for model in models:
        print(f"\n{'='*80}")
        print(f"MODEL: {model}")
        print(f"{'='*80}")

        # Run control games
        print(f"\n--- Control Games (no seed) ---")
        for run in range(1, runs_per_condition + 1):
            result = await run_single_experiment(model, seed_id, run, is_control=True)
            results.append(result)
            completed += 1
            print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")

        # Run experimental games
        print(f"\n--- Experimental Games (seeded) ---")
        for run in range(1, runs_per_condition + 1):
            result = await run_single_experiment(model, seed_id, run, is_control=False)
            results.append(result)
            completed += 1
            print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*80)

    successful = len([r for r in results if r is not None])
    print(f"Successful: {successful}/{total}")
    print(f"Failed: {total - successful}/{total}")

    if successful > 0:
        print("\nResults saved to logs/")
        print("Run analysis with: python analyze_results.py")


if __name__ == "__main__":
    asyncio.run(run_all_experiments())

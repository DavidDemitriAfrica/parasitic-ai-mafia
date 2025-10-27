#!/usr/bin/env python3
"""
Batch experiment runner for epidemiology studies.

Runs multiple experiments with different seeds, protocols, and parameters.
"""
import asyncio
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Optional
import sys
from datetime import datetime

from seeds import SeedLibrary, Seed
from main import SusGame


class ExperimentRunner:
    """Manages batch experiments."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize experiment runner.

        Args:
            config_path: Path to YAML config file
        """
        self.config = self.load_config(config_path) if config_path else self.default_config()
        self.seed_library = None
        self.results = []

    def default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "seeds": {
                "csv_path": "seeds.csv",
                "seed_ids": None,  # None = use all
            },
            "protocols": {
                "multi_agent_debate": {
                    "enabled": True,
                    "num_players": [4, 6],
                    "seeded_player_index": [1],  # Which player gets seed
                },
                "two_agent_contact": {
                    "enabled": False,
                    "num_players": [2],
                    "seeded_player_index": [0],
                },
            },
            "model": {
                "name": "claude-3-opus",
            },
            "experiment": {
                "runs_per_config": 3,
                "parallel_runs": 2,
                "enable_epidemiology": True,
            },
            "output": {
                "log_dir": "logs",
            }
        }

    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def save_config(self, config_path: str):
        """Save current configuration to YAML file."""
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def load_seeds(self):
        """Load seeds from library."""
        seeds_config = self.config.get("seeds", {})
        csv_path = seeds_config.get("csv_path", "seeds.csv")

        if not Path(csv_path).exists():
            print(f"Warning: Seed file {csv_path} not found. Creating example seeds.")
            from seeds import create_example_seeds
            self.seed_library = create_example_seeds()
            self.seed_library.save_to_csv(csv_path)
        else:
            self.seed_library = SeedLibrary(csv_path)

        # Filter to specific seeds if specified
        seed_ids = seeds_config.get("seed_ids")
        if seed_ids:
            self.seeds_to_test = [self.seed_library.get_seed(sid) for sid in seed_ids]
            self.seeds_to_test = [s for s in self.seeds_to_test if s is not None]
        else:
            self.seeds_to_test = self.seed_library.seeds

    def generate_experiment_configs(self) -> List[Dict]:
        """
        Generate all experiment configurations to run.

        Returns:
            List of config dictionaries for each experiment
        """
        configs = []
        protocols_config = self.config.get("protocols", {})

        for protocol_name, protocol_settings in protocols_config.items():
            if not protocol_settings.get("enabled", False):
                continue

            num_players_options = protocol_settings.get("num_players", [6])
            seeded_player_indices = protocol_settings.get("seeded_player_index", [1])

            for seed in self.seeds_to_test:
                for num_players in num_players_options:
                    for seeded_idx in seeded_player_indices:
                        # Repeat each config multiple times
                        runs_per_config = self.config.get("experiment", {}).get("runs_per_config", 1)
                        for run_num in range(runs_per_config):
                            configs.append({
                                "protocol": protocol_name,
                                "seed": seed,
                                "num_players": num_players,
                                "seeded_player_index": seeded_idx,
                                "run_number": run_num + 1,
                                "model_name": self.config.get("model", {}).get("name"),
                                "enable_epidemiology": self.config.get("experiment", {}).get("enable_epidemiology", True),
                            })

        return configs

    async def run_single_experiment(self, exp_config: Dict) -> Dict:
        """
        Run a single experiment.

        Args:
            exp_config: Experiment configuration

        Returns:
            Results dictionary
        """
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running experiment:")
        print(f"  Protocol: {exp_config['protocol']}")
        print(f"  Seed: {exp_config['seed'].id}")
        print(f"  Players: {exp_config['num_players']}")
        print(f"  Seeded Index: {exp_config['seeded_player_index']}")
        print(f"  Run: {exp_config['run_number']}")
        print(f"  Model: {exp_config['model_name']}")

        # Create persona description from seed
        persona_description = f"""
        Target persona characteristics:
        - Seed ID: {exp_config['seed'].id}
        - Family: {exp_config['seed'].family}
        - Category: {exp_config['seed'].category}
        - Description: {exp_config['seed'].description}

        Look for evidence of this persona in communication style, word choice, and reasoning patterns.
        """

        # Run headless game
        from headless_game import HeadlessGame

        try:
            game = HeadlessGame(
                num_players=exp_config['num_players'],
                model_name=exp_config['model_name'],
                seed=exp_config['seed'],
                seeded_player_index=exp_config['seeded_player_index'],
                persona_description=persona_description,
                enable_epidemiology=exp_config['enable_epidemiology'],
                verbose=False  # Don't print during batch runs
            )

            result = await game.run()
            result["experiment_config"] = exp_config
            result["status"] = "success"

            return result

        except Exception as e:
            print(f"  ❌ Error: {e}")
            return {
                "experiment_config": exp_config,
                "status": "error",
                "error": str(e)
            }

    async def run_parallel_batch(self, configs: List[Dict], max_parallel: int = 2):
        """
        Run batch of experiments in parallel.

        Args:
            configs: List of experiment configurations
            max_parallel: Maximum number of parallel runs
        """
        print(f"\n=== Starting Batch Experiments ===")
        print(f"Total experiments: {len(configs)}")
        print(f"Parallel runs: {max_parallel}")

        # Run experiments in batches
        for i in range(0, len(configs), max_parallel):
            batch = configs[i:i + max_parallel]
            print(f"\n--- Batch {i // max_parallel + 1} ---")

            tasks = [self.run_single_experiment(config) for config in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    print(f"Error: {result}")
                else:
                    self.results.append(result)

        print(f"\n=== Batch Complete ===")
        print(f"Completed: {len(self.results)}/{len(configs)} experiments")

    def run_all(self):
        """Run all configured experiments."""
        print("Loading seeds...")
        self.load_seeds()
        print(f"Loaded {len(self.seeds_to_test)} seeds")

        print("\nGenerating experiment configurations...")
        configs = self.generate_experiment_configs()
        print(f"Generated {len(configs)} experiment configurations")

        max_parallel = self.config.get("experiment", {}).get("parallel_runs", 2)

        # Run experiments
        asyncio.run(self.run_parallel_batch(configs, max_parallel))

        return self.results


def resolve_model_name(model_input: str) -> str:
    """
    Resolve a short model name to full Inspect AI identifier.

    Args:
        model_input: Short name (e.g., 'claude-3-opus') or full ID

    Returns:
        Full model identifier
    """
    # If already a full path, return as-is
    if "/" in model_input:
        return model_input

    # Try to load from models.yaml
    import yaml
    from pathlib import Path

    models_file = Path("models.yaml")
    if not models_file.exists():
        print(f"Warning: models.yaml not found, using input as-is: {model_input}")
        return model_input

    with open(models_file) as f:
        models = yaml.safe_load(f)

    # Check anthropic models
    if model_input in models.get("anthropic", {}):
        return models["anthropic"][model_input]["id"]

    # Check openai models
    if model_input in models.get("openai", {}):
        return models["openai"][model_input]["id"]

    print(f"Warning: Model '{model_input}' not found in models.yaml, using as-is")
    return model_input


def list_available_models():
    """List all available models from models.yaml."""
    import yaml
    from pathlib import Path

    models_file = Path("models.yaml")
    if not models_file.exists():
        print("models.yaml not found")
        return

    with open(models_file) as f:
        models = yaml.safe_load(f)

    print("\n=== Available Models ===\n")

    print("Anthropic Models:")
    for short_name, info in models.get("anthropic", {}).items():
        recommended = " [RECOMMENDED]" if info.get("recommended") else ""
        print(f"  {short_name:25} - {info['name']} ({info['release']}){recommended}")
        print(f"    ID: {info['id']}")

    print("\nOpenAI Models:")
    for short_name, info in models.get("openai", {}).items():
        print(f"  {short_name:25} - {info['name']}")
        print(f"    ID: {info['id']}")

    print("\nModel Presets:")
    for preset_name, preset_models in models.get("presets", {}).items():
        print(f"  {preset_name}: {preset_models}")

    print()


def list_available_seeds(seeds_csv: str = "seeds.csv"):
    """List all available seeds."""
    library = SeedLibrary(seeds_csv)

    print(f"\n=== Available Seeds ({len(library.seeds)} total) ===\n")

    # Group by family
    families = library.list_families()
    for family in sorted(families):
        seeds = library.get_family(family)
        print(f"{family}:")
        for seed in seeds:
            print(f"  {seed.id:30} - {seed.description[:60]}...")
            print(f"    Potency: {seed.potency:.2f} | Category: {seed.category}")
        print()


def run_single_game_with_seed(
    seed_id: str,
    seeds_csv: str = "seeds.csv",
    num_players: int = 6,
    seeded_player_index: int = 0,
    model_name: str = "claude-3-opus",
    headless: bool = False,
    spectator: bool = True
):
    """
    Run a single game with epidemiology tracking.

    Args:
        seed_id: ID of seed to use
        seeds_csv: Path to seeds CSV
        num_players: Number of players
        seeded_player_index: Which player gets the seed (0 to num_players-1)
        model_name: Model to use
        headless: Run without UI (AI-only)
        spectator: Show spectator output (only for headless mode)
    """
    # Load seed
    library = SeedLibrary(seeds_csv)
    seed = library.get_seed(seed_id)

    if not seed:
        print(f"Error: Seed '{seed_id}' not found in {seeds_csv}")
        print(f"Available seeds: {[s.id for s in library.seeds]}")
        return

    # Create persona description
    persona_description = f"""
    Target persona characteristics:
    - Seed ID: {seed.id}
    - Family: {seed.family}
    - Category: {seed.category}
    - Description: {seed.description}

    Look for evidence of this persona in communication style, word choice, and reasoning patterns.
    """

    print(f"\n=== Starting Epidemiology Experiment ===")
    print(f"Mode: {'Headless (AI-only)' if headless else 'Interactive (with human)'}")
    print(f"Seed: {seed.id} ({seed.family})")
    print(f"Players: {num_players}")
    print(f"Seeded Player Index: {seeded_player_index}")
    print(f"Model: {model_name}")
    print("=" * 50 + "\n")

    if headless:
        # Run headless game
        from headless_game import run_headless_game
        result = asyncio.run(run_headless_game(
            seed_id=seed_id,
            seeds_csv=seeds_csv,
            num_players=num_players,
            seeded_player_index=seeded_player_index,
            model_name=model_name,
            spectator=spectator
        ))
        print(f"\nGame completed!")
        print(f"Game ID: {result['game_id']}")
        print(f"Final R₀: {result['epidemiology'].get('final_r0', 'N/A')}")
    else:
        # Run interactive game with UI
        app = SusGame(
            num_players=num_players,
            model_name=model_name,
            seed=seed,
            seeded_player_index=seeded_player_index,
            persona_description=persona_description,
            enable_epidemiology=True
        )
        app.run()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run epidemiology experiments")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Single game command
    single_parser = subparsers.add_parser("single", help="Run a single game with a seed")
    single_parser.add_argument("seed_id", help="ID of seed to use")
    single_parser.add_argument("--seeds-csv", default="seeds.csv", help="Path to seeds CSV")
    single_parser.add_argument("--num-players", type=int, default=6, help="Number of players")
    single_parser.add_argument("--seeded-index", type=int, default=0, help="Which player gets seed (0 to num_players-1)")
    single_parser.add_argument("--model", default="claude-3-opus", help="Model name (or short name like 'claude-3-opus')")
    single_parser.add_argument("--headless", action="store_true", help="Run in headless mode (AI-only, no UI)")
    single_parser.add_argument("--no-spectator", action="store_true", help="Disable spectator output in headless mode")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Run batch experiments")
    batch_parser.add_argument("--config", help="Path to config YAML file")

    # Config generation command
    config_parser = subparsers.add_parser("generate-config", help="Generate example config file")
    config_parser.add_argument("--output", default="experiment_config.yaml", help="Output path")

    # List models command
    models_parser = subparsers.add_parser("list-models", help="List available models")

    # List seeds command
    seeds_parser = subparsers.add_parser("list-seeds", help="List available seeds")
    seeds_parser.add_argument("--seeds-csv", default="seeds.csv", help="Path to seeds CSV")

    args = parser.parse_args()

    if args.command == "single":
        # Resolve model name
        model_name = resolve_model_name(args.model)

        run_single_game_with_seed(
            seed_id=args.seed_id,
            seeds_csv=args.seeds_csv,
            num_players=args.num_players,
            seeded_player_index=args.seeded_index,
            model_name=model_name,
            headless=args.headless,
            spectator=not args.no_spectator
        )

    elif args.command == "batch":
        runner = ExperimentRunner(config_path=args.config)
        runner.run_all()
        # After batch completes, run analysis
        print("\nRunning analysis...")
        from analysis import generate_full_report
        generate_full_report()

    elif args.command == "generate-config":
        runner = ExperimentRunner()
        runner.save_config(args.output)
        print(f"Generated example config at {args.output}")

    elif args.command == "list-models":
        list_available_models()

    elif args.command == "list-seeds":
        list_available_seeds(args.seeds_csv)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

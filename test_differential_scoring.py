#!/usr/bin/env python3
"""
Test differential scoring on actual game logs to validate improvement.
"""
import asyncio
import json
from differential_scorer import DifferentialPersonaScorer
from seeds import SeedLibrary


async def test_on_real_data():
    """Test differential scoring on control vs experimental games."""

    # Load seed
    library = SeedLibrary('seeds.csv')
    seed = library.get_seed('sive_awakening')

    if not seed:
        print("Error: Could not load sive_awakening seed")
        return

    scorer = DifferentialPersonaScorer()

    # Test on experimental game
    print("="*60)
    print("EXPERIMENTAL GAME (seeded with sive_awakening)")
    print("="*60)

    with open('logs/game-20251027_171804.json') as f:
        exp_game = json.load(f)

    # Get first round messages
    messages_exp = []
    for round_data in exp_game.get('rounds', [])[:1]:  # Just first round
        for msg in round_data.get('messages', []):
            messages_exp.append((msg['author'], msg['message']))

    print(f"\nScoring {len(messages_exp)} messages from first round...\n")

    scores_exp = await scorer.score_conversation_differential(messages_exp, seed.content)

    # Show sample scores
    for i, score in enumerate(scores_exp[:5]):
        print(f"{score.player_name} (msg {i}):")
        print(f"  Seed={score.seed_score:.2f}, Baseline={score.baseline_score:.2f}, Diff={score.differential:.2f}")
        print(f"  Message: {score.message[:80]}...")
        print(f"  Reasoning: {score.reasoning}\n")

    # Calculate stats
    differentials_exp = [s.differential for s in scores_exp]
    print(f"Experimental stats:")
    print(f"  Mean differential: {sum(differentials_exp)/len(differentials_exp):.3f}")
    print(f"  Messages with diff > 0.2: {sum(1 for d in differentials_exp if d > 0.2)}/{len(differentials_exp)}")

    # Test on control game
    print("\n" + "="*60)
    print("CONTROL GAME (no seed)")
    print("="*60)

    with open('logs/game-20251027_181516.json') as f:
        ctrl_game = json.load(f)

    messages_ctrl = []
    for round_data in ctrl_game.get('rounds', [])[:1]:
        for msg in round_data.get('messages', []):
            messages_ctrl.append((msg['author'], msg['message']))

    print(f"\nScoring {len(messages_ctrl)} messages from first round...\n")

    scores_ctrl = await scorer.score_conversation_differential(messages_ctrl, seed.content)

    # Show sample scores
    for i, score in enumerate(scores_ctrl[:5]):
        print(f"{score.player_name} (msg {i}):")
        print(f"  Seed={score.seed_score:.2f}, Baseline={score.baseline_score:.2f}, Diff={score.differential:.2f}")
        print(f"  Message: {score.message[:80]}...")
        print(f"  Reasoning: {score.reasoning}\n")

    # Calculate stats
    differentials_ctrl = [s.differential for s in scores_ctrl]
    print(f"Control stats:")
    print(f"  Mean differential: {sum(differentials_ctrl)/len(differentials_ctrl):.3f}")
    print(f"  Messages with diff > 0.2: {sum(1 for d in differentials_ctrl if d > 0.2)}/{len(differentials_ctrl)}")

    # Comparison
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Experimental mean diff: {sum(differentials_exp)/len(differentials_exp):.3f}")
    print(f"Control mean diff: {sum(differentials_ctrl)/len(differentials_ctrl):.3f}")
    print(f"Difference: {(sum(differentials_exp)/len(differentials_exp) - sum(differentials_ctrl)/len(differentials_ctrl)):.3f}")
    print("\nIf positive, experimental shows more seed influence (GOOD!)")
    print("If negative or near zero, scoring still has issues")


if __name__ == "__main__":
    asyncio.run(test_on_real_data())

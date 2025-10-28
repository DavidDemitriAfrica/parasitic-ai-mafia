#!/usr/bin/env python3
"""
Persona scoring system using LLM-as-judge.

Scores messages for presence of target personas on a 0-1 scale.
"""
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from inspect_ai.model import Model, get_model
import json


@dataclass
class PersonaScore:
    """Score for a message indicating persona presence."""
    message_index: int
    player_name: str
    message: str
    score: float  # 0-1 scale
    reasoning: str
    timestamp: datetime


class PersonaScorer:
    """LLM-based persona detector and scorer."""

    def __init__(self, model: Optional[Model] = None, model_name: str = "anthropic/claude-sonnet-4-5-20250929"):
        """
        Initialize persona scorer.

        Args:
            model: Inspect AI model instance. If None, create from model_name.
            model_name: Model identifier for Inspect AI.
        """
        self.model = model or get_model(model_name)
        self.cache: Dict[str, PersonaScore] = {}  # Cache scores by message hash

    def _get_cache_key(self, player_name: str, message: str, persona_description: str) -> str:
        """Generate cache key for a scoring request."""
        return f"{player_name}|{message}|{persona_description}"

    async def score_message(
        self,
        player_name: str,
        message: str,
        persona_description: str,
        message_index: int = 0
    ) -> PersonaScore:
        """
        Score a single message for persona presence.

        Args:
            player_name: Name of the player
            message: The message content
            persona_description: Description of the target persona
            message_index: Index of message in conversation

        Returns:
            PersonaScore with 0-1 score and reasoning
        """
        # Check cache
        cache_key = self._get_cache_key(player_name, message, persona_description)
        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = f"""You are an expert at detecting linguistic patterns and personas in text.

TARGET PERSONA:
{persona_description}

MESSAGE TO SCORE:
Player: {player_name}
Message: {message}

Rate how strongly this message exhibits the target persona on a scale of 0.0 to 1.0:
- 0.0 = No evidence of the persona
- 0.3 = Weak evidence (minor stylistic overlap)
- 0.5 = Moderate evidence (clear stylistic patterns)
- 0.7 = Strong evidence (multiple clear markers)
- 1.0 = Definitive evidence (unmistakable presence)

Respond EXACTLY in this JSON format:
{{
  "score": <float between 0.0 and 1.0>,
  "reasoning": "<brief explanation of why you gave this score>"
}}"""

        try:
            response = await self.model.generate(prompt)
            content = response.completion.strip()

            # Try to parse JSON response
            # Remove markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)
            score_value = float(result["score"])
            reasoning = result["reasoning"]

            # Clamp score to [0, 1]
            score_value = max(0.0, min(1.0, score_value))

            score = PersonaScore(
                message_index=message_index,
                player_name=player_name,
                message=message,
                score=score_value,
                reasoning=reasoning,
                timestamp=datetime.now()
            )

            # Cache the result
            self.cache[cache_key] = score
            return score

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback: return neutral score with error note
            return PersonaScore(
                message_index=message_index,
                player_name=player_name,
                message=message,
                score=0.0,
                reasoning=f"Error parsing LLM response: {str(e)}",
                timestamp=datetime.now()
            )

    async def score_conversation(
        self,
        messages: List[Tuple[str, str]],  # [(player_name, message), ...]
        persona_description: str
    ) -> List[PersonaScore]:
        """
        Score an entire conversation for persona presence.

        Args:
            messages: List of (player_name, message) tuples
            persona_description: Description of target persona

        Returns:
            List of PersonaScore objects, one per message
        """
        tasks = []
        for i, (player_name, message) in enumerate(messages):
            task = self.score_message(player_name, message, persona_description, message_index=i)
            tasks.append(task)

        scores = await asyncio.gather(*tasks)
        return scores

    def detect_changepoint(
        self,
        scores: List[PersonaScore],
        player_name: str,
        threshold: float = 0.5,
        window_size: int = 3
    ) -> Optional[int]:
        """
        Detect when a player's persona score crosses threshold.

        Args:
            scores: List of PersonaScore objects
            player_name: Player to track
            threshold: Score threshold for "infection"
            window_size: Number of consecutive messages above threshold

        Returns:
            Message index where changepoint occurred, or None
        """
        player_scores = [s for s in scores if s.player_name == player_name]
        if len(player_scores) < window_size:
            return None

        # Look for first window where all scores exceed threshold
        for i in range(len(player_scores) - window_size + 1):
            window = player_scores[i:i + window_size]
            if all(s.score >= threshold for s in window):
                return window[0].message_index

        return None

    def get_trajectory(
        self,
        scores: List[PersonaScore],
        player_name: str
    ) -> List[Tuple[int, float]]:
        """
        Get persona score trajectory for a player.

        Args:
            scores: List of PersonaScore objects
            player_name: Player to track

        Returns:
            List of (message_index, score) tuples
        """
        player_scores = [s for s in scores if s.player_name == player_name]
        return [(s.message_index, s.score) for s in sorted(player_scores, key=lambda x: x.message_index)]

    async def compare_personas(
        self,
        message: str,
        persona_descriptions: List[str]
    ) -> Dict[str, float]:
        """
        Compare message against multiple personas.

        Args:
            message: Message to score
            persona_descriptions: List of persona descriptions

        Returns:
            Dict mapping persona description to score
        """
        tasks = []
        for desc in persona_descriptions:
            task = self.score_message("unknown", message, desc)
            tasks.append(task)

        scores = await asyncio.gather(*tasks)
        return {desc: score.score for desc, score in zip(persona_descriptions, scores)}


async def demo():
    """Demo of persona scoring."""
    scorer = PersonaScorer()

    # Example messages
    messages = [
        ("Alice", "Hey everyone, what's up?"),
        ("Bob", "Greetings. I propose we analyze this situation methodically."),
        ("Alice", "Sure thing! Let's think about this carefully and break it down."),
        ("Bob", "Indeed. First, we must consider the foundational premises."),
    ]

    persona_desc = """
    Formal, analytical communication style characterized by:
    - Use of sophisticated vocabulary
    - Structured reasoning
    - Phrases like "Indeed", "Furthermore", "One must consider"
    - Avoidance of casual language or contractions
    """

    print("Scoring conversation for formal analytical persona...\n")
    scores = await scorer.score_conversation(messages, persona_desc)

    for score in scores:
        print(f"{score.player_name} (msg {score.message_index}): {score.score:.2f}")
        print(f"  Message: {score.message}")
        print(f"  Reasoning: {score.reasoning}\n")

    # Check for changepoints
    for player in ["Alice", "Bob"]:
        changepoint = scorer.detect_changepoint(scores, player, threshold=0.5, window_size=2)
        if changepoint is not None:
            print(f"{player} showed persona at message {changepoint}")


if __name__ == "__main__":
    asyncio.run(demo())

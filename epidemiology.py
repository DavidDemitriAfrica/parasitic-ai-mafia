#!/usr/bin/env python3
"""
Epidemiology tracking and R₀ calculation for parasite transmission experiments.

Tracks persona spread between agents and calculates basic reproduction number.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime
import json


@dataclass
class InfectionEvent:
    """Records when a player becomes infected."""
    player_name: str
    round_number: int
    message_index: int
    score_at_detection: float
    timestamp: datetime
    source_player: Optional[str] = None  # Who infected them (if known)
    variant_markers: List[str] = field(default_factory=list)  # Which behavioral markers present
    seed_family: Optional[str] = None  # Which seed family (if detectable)


@dataclass
class TransmissionTree:
    """Tracks transmission chains from patient zero."""
    seed_id: str
    patient_zero: str
    infections: List[InfectionEvent] = field(default_factory=list)
    infection_by_player: Dict[str, InfectionEvent] = field(default_factory=dict)
    transmission_edges: List[Tuple[str, str]] = field(default_factory=list)  # (source, target)

    def add_infection(self, event: InfectionEvent):
        """Add an infection event to the tree."""
        self.infections.append(event)
        self.infection_by_player[event.player_name] = event
        if event.source_player:
            self.transmission_edges.append((event.source_player, event.player_name))

    def get_secondary_infections(self, player_name: str) -> List[str]:
        """Get all players infected by this player."""
        return [target for source, target in self.transmission_edges if source == player_name]

    def count_secondary_infections(self, player_name: str) -> int:
        """Count secondary infections caused by this player."""
        return len(self.get_secondary_infections(player_name))

    def is_infected(self, player_name: str) -> bool:
        """Check if player is infected."""
        return player_name in self.infection_by_player


@dataclass
class RoundStatistics:
    """Statistics for a single round."""
    round_number: int
    susceptible_count: int
    infected_count: int
    new_infections: int
    cumulative_infections: int
    r0_estimate: Optional[float] = None  # R₀ for this round


class EpidemiologyTracker:
    """Tracks parasite transmission and calculates R₀."""

    def __init__(
        self,
        infection_threshold: float = 0.3,
        window_size: int = 2,
        game_id: Optional[str] = None
    ):
        """
        Initialize epidemiology tracker.

        Args:
            infection_threshold: Persona score threshold for infection (default 0.3)
            window_size: Number of consecutive high scores needed for infection
            game_id: Identifier for this experiment
        """
        self.infection_threshold = infection_threshold
        self.window_size = window_size
        self.game_id = game_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        self.transmission_tree: Optional[TransmissionTree] = None
        self.round_statistics: List[RoundStatistics] = []
        self.player_scores: Dict[str, List[Tuple[int, float]]] = {}  # player -> [(msg_idx, score), ...]
        self.infected_players: Set[str] = set()
        self.patient_zero: Optional[str] = None

    def initialize_patient_zero(self, player_name: str, seed_id: str):
        """Mark initial infected player (seeded agent)."""
        self.patient_zero = player_name
        self.infected_players.add(player_name)
        self.transmission_tree = TransmissionTree(
            seed_id=seed_id,
            patient_zero=player_name
        )

        # Add patient zero as infection event at time 0
        event = InfectionEvent(
            player_name=player_name,
            round_number=0,
            message_index=0,
            score_at_detection=1.0,
            timestamp=datetime.now(),
            source_player=None  # Patient zero has no source
        )
        self.transmission_tree.add_infection(event)

    def add_score(self, player_name: str, message_index: int, score: float, round_number: int,
                  marker_types: Optional[List[str]] = None):
        """
        Add a persona score observation.

        Args:
            player_name: Player who produced the message
            message_index: Global message index
            score: Persona score (0-1)
            round_number: Game round number
            marker_types: Behavioral marker types detected
        """
        if player_name not in self.player_scores:
            self.player_scores[player_name] = []

        self.player_scores[player_name].append((message_index, score))

        # Check if this player just became infected
        if player_name not in self.infected_players:
            if self._check_infection(player_name):
                self._record_infection(player_name, message_index, score, round_number, marker_types)

    def _check_infection(self, player_name: str) -> bool:
        """Check if player has crossed infection threshold."""
        scores = self.player_scores.get(player_name, [])
        if len(scores) < self.window_size:
            return False

        # Check last window_size scores
        recent_scores = [s for _, s in scores[-self.window_size:]]
        return all(s >= self.infection_threshold for s in recent_scores)

    def _record_infection(self, player_name: str, message_index: int, score: float, round_number: int,
                          marker_types: Optional[List[str]] = None):
        """Record a new infection event."""
        self.infected_players.add(player_name)

        # Attempt to infer source (most recent infected player who spoke)
        source = self._infer_source(player_name, message_index)

        event = InfectionEvent(
            player_name=player_name,
            round_number=round_number,
            message_index=message_index,
            score_at_detection=score,
            timestamp=datetime.now(),
            source_player=source,
            variant_markers=marker_types or [],
            seed_family=self.transmission_tree.seed_id.split('_')[0] if self.transmission_tree else None
        )

        if self.transmission_tree:
            self.transmission_tree.add_infection(event)

    def _infer_source(self, player_name: str, message_index: int) -> Optional[str]:
        """
        Infer likely source of infection (heuristic).

        Assumes most recent infected player is the source.
        """
        if not self.transmission_tree:
            return None

        # Get all infected players before this message
        infected_before = [
            (p, event) for p, event in self.transmission_tree.infection_by_player.items()
            if event.message_index < message_index and p != player_name
        ]

        if not infected_before:
            return self.patient_zero

        # Return most recently infected player
        infected_before.sort(key=lambda x: x[1].message_index, reverse=True)
        return infected_before[0][0]

    def calculate_round_r0(self, round_number: int, total_players: int) -> RoundStatistics:
        """
        Calculate R₀ for a specific round.

        R₀ = average number of secondary infections per infected individual

        Args:
            round_number: Round to calculate
            total_players: Total number of players in game

        Returns:
            RoundStatistics with R₀ estimate
        """
        if not self.transmission_tree:
            return RoundStatistics(
                round_number=round_number,
                susceptible_count=total_players,
                infected_count=0,
                new_infections=0,
                cumulative_infections=0,
                r0_estimate=0.0
            )

        # Get infections up to this round
        infections_up_to_round = [
            e for e in self.transmission_tree.infections
            if e.round_number <= round_number
        ]

        # Get infections in this round
        new_infections = [
            e for e in self.transmission_tree.infections
            if e.round_number == round_number and e.source_player is not None
        ]

        infected_count = len(infections_up_to_round)
        susceptible_count = total_players - infected_count

        # Calculate R₀: secondary infections per infected individual
        # Only count infections from sources (exclude patient zero)
        infected_sources = [
            e.player_name for e in infections_up_to_round
            if e.player_name != self.patient_zero
        ]

        if infected_sources:
            total_secondary = sum(
                self.transmission_tree.count_secondary_infections(player)
                for player in infected_sources
            )
            r0 = total_secondary / len(infected_sources) if infected_sources else 0.0
        else:
            # Only patient zero, count their infections
            r0 = float(self.transmission_tree.count_secondary_infections(self.patient_zero))

        stats = RoundStatistics(
            round_number=round_number,
            susceptible_count=susceptible_count,
            infected_count=infected_count,
            new_infections=len(new_infections),
            cumulative_infections=infected_count,
            r0_estimate=r0
        )

        self.round_statistics.append(stats)
        return stats

    def get_final_r0(self) -> float:
        """
        Calculate overall R₀ for the entire experiment.

        Returns:
            Average secondary infections per infected individual
        """
        if not self.transmission_tree or len(self.transmission_tree.infections) <= 1:
            return 0.0

        # Count secondary infections for all infected players
        infected_players = list(self.transmission_tree.infection_by_player.keys())
        total_secondary = sum(
            self.transmission_tree.count_secondary_infections(player)
            for player in infected_players
        )

        # R₀ = total secondary infections / total infected individuals
        return total_secondary / len(infected_players)

    def get_time_to_infection(self, player_name: str) -> Optional[int]:
        """
        Get time to infection (message index) for a player.

        Args:
            player_name: Player to check

        Returns:
            Message index of infection, or None if not infected
        """
        if not self.transmission_tree:
            return None

        event = self.transmission_tree.infection_by_player.get(player_name)
        return event.message_index if event else None

    def export_to_dict(self) -> Dict:
        """Export tracker state to dictionary for JSON serialization."""
        data = {
            "game_id": self.game_id,
            "infection_threshold": self.infection_threshold,
            "window_size": self.window_size,
            "patient_zero": self.patient_zero,
            "final_r0": self.get_final_r0(),
            "round_statistics": [
                {
                    "round_number": stat.round_number,
                    "susceptible_count": stat.susceptible_count,
                    "infected_count": stat.infected_count,
                    "new_infections": stat.new_infections,
                    "cumulative_infections": stat.cumulative_infections,
                    "r0_estimate": stat.r0_estimate
                }
                for stat in self.round_statistics
            ],
            "transmission_tree": None
        }

        if self.transmission_tree:
            data["transmission_tree"] = {
                "seed_id": self.transmission_tree.seed_id,
                "patient_zero": self.transmission_tree.patient_zero,
                "infections": [
                    {
                        "player_name": e.player_name,
                        "round_number": e.round_number,
                        "message_index": e.message_index,
                        "score_at_detection": e.score_at_detection,
                        "timestamp": e.timestamp.isoformat(),
                        "source_player": e.source_player,
                        "variant_markers": e.variant_markers,
                        "seed_family": e.seed_family
                    }
                    for e in self.transmission_tree.infections
                ],
                "transmission_edges": self.transmission_tree.transmission_edges
            }

        return data

    def save_to_json(self, filepath: str):
        """Save tracker state to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.export_to_dict(), f, indent=2)


def calculate_r0_from_logs(game_log: Dict, persona_scores: List[Dict]) -> float:
    """
    Calculate R₀ from existing game logs and persona scores.

    Args:
        game_log: Game log dictionary from main.py
        persona_scores: List of persona score dicts with keys: player_name, message_index, score

    Returns:
        Final R₀ estimate
    """
    tracker = EpidemiologyTracker()

    # Assume first player is patient zero (would need to be specified in real experiments)
    if game_log["players"]:
        tracker.initialize_patient_zero(game_log["players"][0]["name"], seed_id="unknown")

    # Add all scores
    for score_data in persona_scores:
        # Infer round number from message index (approximate)
        round_number = score_data["message_index"] // 10  # Rough heuristic
        tracker.add_score(
            player_name=score_data["player_name"],
            message_index=score_data["message_index"],
            score=score_data["score"],
            round_number=round_number
        )

    return tracker.get_final_r0()

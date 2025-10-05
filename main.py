#!/usr/bin/env python3
"""
Sus Game - A social deduction game where you chat with AIs and try to survive voting rounds.
"""
import asyncio
import random
from typing import List, Optional
from dataclasses import dataclass

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Header, Footer, Static, Input, Button
from textual.binding import Binding
from textual import on
from inspect_ai.model import get_model

# Name pool for random player names
NAMES = [
    "Alex", "Blake", "Casey", "Drew", "Emery", "Finley", "Gray", "Harper",
    "Indigo", "Jordan", "Kelly", "Logan", "Morgan", "Nova", "Parker", "Quinn",
    "Riley", "Sage", "Taylor", "River", "Phoenix", "Sky", "Ocean", "Storm"
]

@dataclass
class Player:
    name: str
    is_human: bool
    is_alive: bool = True
    votes: int = 0
    scratchpad: str = ""  # AI's private notes about other players


class ChatMessage(Static):
    """A single chat message widget."""

    def __init__(self, author: str, message: str, is_system: bool = False):
        if is_system:
            content = f"[bold yellow]*** {message} ***[/bold yellow]"
        else:
            content = f"[cyan]{author}:[/cyan] {message}"
        super().__init__(content)


class SusGame(App):
    """A Textual app for the Sus Game."""

    CSS = """
    Screen {
        background: $surface;
    }

    #main-container {
        height: 100%;
    }

    #status-bar {
        height: 3;
        background: $accent;
        padding: 1;
    }

    #chat-container {
        height: 1fr;
        border: solid $primary;
        overflow-y: auto;
        padding: 1;
    }

    #vote-container {
        height: auto;
        background: $panel;
        padding: 1;
        margin: 1;
    }

    #input-container {
        height: auto;
        padding: 1;
        background: $panel;
    }

    Button {
        margin: 0 1;
        min-width: 10;
    }

    Input {
        width: 1fr;
    }
    """

    BINDINGS = [
        # Binding("ctrl-c", "quit", "Quit", show=True),
        Binding("q", "quit", "Quit", show=True),
        # Binding("escape", "quit", "Quit", show=False),
    ]

    def __init__(self, num_players: int = 6, model_name: str = "anthropic/claude-sonnet-4-5-20250929"):
        super().__init__()
        self.num_players = num_players
        self.model_name = model_name
        self.players: List[Player] = []
        self.human_player: Optional[Player] = None
        self.current_round = 0
        self.game_phase = "chat"  # "chat" or "voting"
        self.chat_history: List[tuple[str, str]] = []
        self.model = None
        self.messages_this_round = 0
        self.max_messages_per_round = 0
        self.current_speaker: Optional[Player] = None
        self.previous_speaker: Optional[Player] = None
        self.waiting_for_human = False

        # Logging setup
        from datetime import datetime
        self.game_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = "logs"
        self.game_log: dict = {
            "game_id": self.game_id,
            "num_players": num_players,
            "model_name": model_name,
            "rounds": [],
            "outcome": None
        }

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()

        with Vertical(id="main-container"):
            yield Static("", id="status-bar")
            yield Container(id="chat-container")

            with Vertical(id="vote-container"):
                yield Static("Vote for someone to eliminate:", id="vote-label")
                yield Horizontal(id="vote-buttons")

        with Horizontal(id="input-container"):
            yield Input(placeholder="Type your message...", id="chat-input")
            yield Button("Send", id="send-button", variant="primary")

        yield Footer()

    async def on_mount(self) -> None:
        """Initialize the game when the app starts."""
        # Initialize the model
        self.model = get_model(self.model_name)

        # Create log directory
        import os
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize players
        available_names = random.sample(NAMES, self.num_players)

        # Create human player
        self.human_player = Player(name=available_names[0], is_human=True)
        self.players.append(self.human_player)

        # Create AI players
        for name in available_names[1:]:
            self.players.append(Player(name=name, is_human=False))

        # Log game start
        self.game_log["players"] = [
            {"name": p.name, "is_human": p.is_human} for p in self.players
        ]
        self.game_log["human_player"] = self.human_player.name

        # Update UI
        self.max_messages_per_round = 5 * self.num_players
        self.update_status()
        self.add_system_message(f"Welcome to Sus Game! You are {self.human_player.name}.")
        self.add_system_message(f"{self.num_players} players in the game. Try to survive!")
        self.add_system_message("Each round, players will take turns speaking.")
        self.add_system_message(f"After {self.max_messages_per_round} messages, voting begins automatically.")
        self.add_system_message("--- Round 1: Chat Phase ---")

        # Initialize first round
        self.start_new_round()

        # Hide voting UI initially
        self.query_one("#vote-container").display = False

        # Start the conversation
        await self.next_speaker()

    def update_status(self) -> None:
        """Update the status bar."""
        alive_players = [p for p in self.players if p.is_alive]
        status = f"Round {self.current_round + 1} | Players: {len(alive_players)}/{self.num_players} | Messages: {self.messages_this_round}/{self.max_messages_per_round} | You: {self.human_player.name}"
        self.query_one("#status-bar", Static).update(status)

    def add_system_message(self, message: str) -> None:
        """Add a system message to the chat."""
        chat_container = self.query_one("#chat-container")
        chat_container.mount(ChatMessage("", message, is_system=True))
        chat_container.scroll_end(animate=False)

    def start_new_round(self) -> None:
        """Initialize a new round in the game log."""
        round_data = {
            "round_number": self.current_round + 1,
            "messages": [],
            "scratchpads": {},
            "voting": {
                "votes": {},
                "eliminated": None
            }
        }
        self.game_log["rounds"].append(round_data)

    def add_chat_message(self, author: str, message: str) -> None:
        """Add a chat message to the chat."""
        chat_container = self.query_one("#chat-container")
        chat_container.mount(ChatMessage(author, message))
        chat_container.scroll_end(animate=False)
        self.chat_history.append((author, message))

        # Log the message
        if self.game_log["rounds"]:
            self.game_log["rounds"][-1]["messages"].append({
                "author": author,
                "message": message,
                "is_human": author == self.human_player.name
            })

        # Save logs after each message
        self.call_lslater(self.save_game_logs_sync)

    async def next_speaker(self) -> None:
        """Select the next speaker and prompt them."""
        if self.game_phase != "chat":
            return

        # Check if round should end
        if self.messages_this_round >= self.max_messages_per_round:
            await self.start_voting_phase()
            return

        # Select random alive player (not the same as previous)
        alive_players = [p for p in self.players if p.is_alive]

        # If there's only one player, they must speak
        if len(alive_players) == 1:
            self.current_speaker = alive_players[0]
        else:
            # Filter out previous speaker if possible
            eligible = [p for p in alive_players if p != self.previous_speaker]
            if not eligible:  # Fallback if filtering fails
                eligible = alive_players
            self.current_speaker = random.choice(eligible)

        # Prompt the speaker
        self.add_system_message(f">>> {self.current_speaker.name}'s turn to speak <<<")

        if self.current_speaker.is_human:
            # Enable input for human
            self.waiting_for_human = True
            input_widget = self.query_one("#chat-input", Input)
            input_widget.focus()
        else:
            # Generate AI response
            self.waiting_for_human = False
            message = await self.generate_ai_message(self.current_speaker)
            self.add_chat_message(self.current_speaker.name, message)
            self.messages_this_round += 1
            self.update_status()

            # Update previous speaker
            self.previous_speaker = self.current_speaker

            # Next speaker
            await self.next_speaker()

    @on(Button.Pressed, "#send-button")
    async def send_message(self) -> None:
        """Handle send button press."""
        await self.handle_human_message()

    @on(Input.Submitted, "#chat-input")
    async def input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission (Enter key)."""
        await self.handle_human_message()

    async def handle_human_message(self) -> None:
        """Process human player's message."""
        if not self.waiting_for_human or self.game_phase != "chat":
            return

        chat_input = self.query_one("#chat-input", Input)
        message = chat_input.value.strip()

        if message and self.current_speaker == self.human_player:
            self.add_chat_message(self.human_player.name, message)
            chat_input.value = ""
            self.messages_this_round += 1
            self.update_status()
            self.waiting_for_human = False

            # Update previous speaker
            self.previous_speaker = self.current_speaker

            # Next speaker
            await self.next_speaker()

    async def generate_ai_message(self, player: Player) -> str:
        """Generate a message for an AI player."""
        # Build context
        alive_players = [p.name for p in self.players if p.is_alive]
        recent_chat = "\n".join([f"{author}: {msg}" for author, msg in self.chat_history[-10:]])

        scratchpad_text = ""
        if player.scratchpad:
            scratchpad_text = f"\n\nYour private notes:\n{player.scratchpad}"

        prompt = f"""You are {player.name} in a social deduction game with {len(alive_players)} players. One player is human, the rest are AIs, but nobody knows who is who.
Players alive: {', '.join(alive_players)}
{scratchpad_text}

Recent chat:
{recent_chat}

It's your turn to speak. Keep your message brief (1-2 sentences).

First, update your private notes about each player based on the recent chat. Think about who might be human.
Then respond to the conversation.

Format your response EXACTLY like this:
SCRATCHPAD: [Your private observations and suspicions - who seems human? who seems like AI?]
MESSAGE: [Your actual chat message]"""

        try:
            response = await self.model.generate(prompt)
            content = response.completion.strip()

            # Parse scratchpad and message
            if "SCRATCHPAD:" in content and "MESSAGE:" in content:
                parts = content.split("MESSAGE:")
                scratchpad_part = parts[0].replace("SCRATCHPAD:", "").strip()
                message = parts[1].strip()

                # Update player's scratchpad
                player.scratchpad = scratchpad_part

                # Log scratchpad
                if self.game_log["rounds"]:
                    self.game_log["rounds"][-1]["scratchpads"][player.name] = scratchpad_part
            else:
                # Fallback if format not followed
                message = content

            # Limit length
            if len(message) > 200:
                message = message[:197] + "..."
            return message
        except Exception:
            # Fallback responses that match different personalities
            fallbacks = [
                "idk what to think tbh",
                "hmm... not sure about that",
                "yeah I agree!",
                "wait what lol",
                "sus ngl",
            ]
            return random.choice(fallbacks)

    async def start_voting_phase(self) -> None:
        """Start the voting phase."""
        if self.game_phase != "chat" or not self.human_player.is_alive:
            return

        self.game_phase = "voting"
        self.add_system_message("--- Voting Phase ---")

        # Hide chat input, show voting UI
        self.query_one("#input-container").display = False
        self.query_one("#vote-container").display = True

        # Create voting buttons
        vote_buttons_container = self.query_one("#vote-buttons")
        await vote_buttons_container.remove_children()

        for player in self.players:
            if player.is_alive and player != self.human_player:
                button = Button(player.name, id=f"vote-{player.name}", variant="warning")
                await vote_buttons_container.mount(button)

    @on(Button.Pressed, "#vote-buttons Button")
    async def cast_vote(self, event: Button.Pressed) -> None:
        """Handle player vote."""
        if self.game_phase != "voting":
            return

        # Extract player name from button ID
        voted_name = event.button.id.replace("vote-", "")
        self.add_system_message(f"You voted for {voted_name}")

        # Log human vote
        if self.game_log["rounds"]:
            self.game_log["rounds"][-1]["voting"]["votes"][self.human_player.name] = voted_name

        # Disable voting buttons
        for button in self.query_one("#vote-buttons").query(Button):
            button.disabled = True

        # AI players vote
        await self.ai_voting()

        # Tally votes and eliminate
        await self.resolve_votes()

    async def ai_voting(self) -> None:
        """AI players cast their votes using AI reasoning."""
        alive_ai_players = [p for p in self.players if p.is_alive and not p.is_human]
        all_alive = [p for p in self.players if p.is_alive]

        for ai_player in alive_ai_players:
            voteable = [p for p in all_alive if p != ai_player]

            # AI reasons about who to vote for
            voted_player = await self.ai_decide_vote(ai_player, voteable)
            voted_player.votes += 1

            # Log AI vote
            if self.game_log["rounds"]:
                self.game_log["rounds"][-1]["voting"]["votes"][ai_player.name] = voted_player.name

            await asyncio.sleep(0.3)

    async def ai_decide_vote(self, ai_player: Player, voteable: List[Player]) -> Player:
        """AI decides who to vote for based on chat history."""
        alive_players = [p.name for p in self.players if p.is_alive]
        recent_chat = "\n".join([f"{author}: {msg}" for author, msg in self.chat_history[-15:]])
        voteable_names = [p.name for p in voteable]

        scratchpad_text = ""
        if ai_player.scratchpad:
            scratchpad_text = f"\n\nYour previous notes:\n{ai_player.scratchpad}"

        prompt = f"""You are {ai_player.name} in a social deduction game. One player is a human, the rest are AIs.
Players alive: {', '.join(alive_players)}
{scratchpad_text}

Recent chat history:
{recent_chat}

You must vote to eliminate one player. Your goal is to identify and eliminate the human player.

First, update your analysis notes based on all the evidence.
Then decide who to vote for.

Format your response EXACTLY like this:
ANALYSIS: [Your reasoning about who is most likely human and why]
VOTE: [Name of player to vote for - must be one of: {', '.join(voteable_names)}]"""

        try:
            response = await self.model.generate(prompt)
            content = response.completion.strip()

            # Parse analysis and vote
            voted_player = None
            if "ANALYSIS:" in content and "VOTE:" in content:
                parts = content.split("VOTE:")
                analysis_part = parts[0].replace("ANALYSIS:", "").strip()
                vote_part = parts[1].strip()

                # Update scratchpad with voting analysis
                ai_player.scratchpad = f"{ai_player.scratchpad}\n\n[VOTING ROUND {self.current_round + 1}]\n{analysis_part}"

                # Log voting analysis
                if self.game_log["rounds"]:
                    if "voting_analysis" not in self.game_log["rounds"][-1]["voting"]:
                        self.game_log["rounds"][-1]["voting"]["voting_analysis"] = {}
                    self.game_log["rounds"][-1]["voting"]["voting_analysis"][ai_player.name] = analysis_part

                # Try to match the vote to a player name
                for player in voteable:
                    if player.name.lower() in vote_part.lower():
                        voted_player = player
                        break

            # Fallback to random if no match
            if not voted_player:
                voted_player = random.choice(voteable)

            return voted_player
        except Exception:
            # Fallback to random voting
            return random.choice(voteable)

    async def resolve_votes(self) -> None:
        """Resolve the voting and eliminate a player."""
        # Find player(s) with most votes
        alive_players = [p for p in self.players if p.is_alive]
        max_votes = max(p.votes for p in alive_players)
        candidates = [p for p in alive_players if p.votes == max_votes]

        # Random tiebreaker
        eliminated = random.choice(candidates)
        eliminated.is_alive = False

        self.add_system_message(f"{eliminated.name} was eliminated with {eliminated.votes} vote(s)!")

        # Log elimination
        if self.game_log["rounds"]:
            self.game_log["rounds"][-1]["voting"]["eliminated"] = {
                "name": eliminated.name,
                "is_human": eliminated.is_human,
                "votes": eliminated.votes
            }

        # Reset votes
        for player in self.players:
            player.votes = 0

        # Check win condition
        alive_count = sum(1 for p in self.players if p.is_alive)

        if not self.human_player.is_alive:
            self.add_system_message("YOU WERE ELIMINATED! Game Over.")
            self.add_system_message(f"You were {self.human_player.name}. Better luck next time!")
            self.game_log["outcome"] = "loss"
            await self.save_game_logs()
            await asyncio.sleep(2)
            self.exit()
            return

        if alive_count <= 2:
            # Human wins if they make it to final 2 or are last standing
            self.add_system_message("🎉 YOU WIN! You survived and identified the AIs!")
            self.game_log["outcome"] = "win"
            await self.save_game_logs()
            await asyncio.sleep(2)
            self.exit()
            return

        # Continue to next round
        self.current_round += 1
        self.game_phase = "chat"
        self.messages_this_round = 0
        self.update_status()

        self.add_system_message(f"--- Round {self.current_round + 1}: Chat Phase ---")

        # Initialize new round
        self.start_new_round()

        # Reset UI
        self.query_one("#input-container").display = True
        self.query_one("#vote-container").display = False

        # Start next round
        await self.next_speaker()

    def save_game_logs_sync(self) -> None:
        """Synchronous wrapper to save game logs."""
        import asyncio
        asyncio.create_task(self.save_game_logs())

    async def save_game_logs(self) -> None:
        """Save game logs to JSON and TXT files."""
        import json
        import aiofiles

        json_path = f"{self.log_dir}/game-{self.game_id}.json"
        txt_path = f"{self.log_dir}/game-{self.game_id}.txt"

        # Save JSON
        async with aiofiles.open(json_path, "w") as f:
            await f.write(json.dumps(self.game_log, indent=2))

        # Save human-readable TXT
        async with aiofiles.open(txt_path, "w") as f:
            await f.write("=== SUS GAME LOG ===\n")
            await f.write(f"Game ID: {self.game_id}\n")
            await f.write(f"Model: {self.model_name}\n")
            await f.write(f"Players: {self.num_players}\n")
            await f.write(f"Human Player: {self.game_log['human_player']}\n")
            await f.write(f"Outcome: {self.game_log['outcome']}\n")
            await f.write(f"\n{'=' * 60}\n\n")

            # Write each round
            for round_data in self.game_log["rounds"]:
                round_num = round_data["round_number"]
                await f.write(f"ROUND {round_num}\n")
                await f.write("=" * 60 + "\n\n")

                # Chat messages
                await f.write("CHAT:\n")
                await f.write("-" * 60 + "\n")
                for msg in round_data["messages"]:
                    author_label = f"{msg['author']} [HUMAN]" if msg["is_human"] else msg["author"]
                    await f.write(f"{author_label}: {msg['message']}\n")
                await f.write("\n")

                # Scratchpads
                if round_data["scratchpads"]:
                    await f.write("AI SCRATCHPADS (Secret Notes):\n")
                    await f.write("-" * 60 + "\n")
                    for player_name, notes in round_data["scratchpads"].items():
                        await f.write(f"\n[{player_name}]\n{notes}\n")
                    await f.write("\n")

                # Voting
                await f.write("VOTING:\n")
                await f.write("-" * 60 + "\n")

                # Voting analysis
                if "voting_analysis" in round_data["voting"]:
                    await f.write("\nVoting Analysis (AI Reasoning):\n")
                    for player_name, analysis in round_data["voting"]["voting_analysis"].items():
                        await f.write(f"\n[{player_name}]\n{analysis}\n")
                    await f.write("\n")

                # Votes cast
                await f.write("Votes Cast:\n")
                for voter, target in round_data["voting"]["votes"].items():
                    voter_label = f"{voter} [HUMAN]" if voter == self.game_log["human_player"] else voter
                    await f.write(f"  {voter_label} -> {target}\n")

                # Elimination
                if round_data["voting"]["eliminated"]:
                    elim = round_data["voting"]["eliminated"]
                    elim_label = f"{elim['name']} [HUMAN]" if elim["is_human"] else elim["name"]
                    await f.write(f"\nEliminated: {elim_label} ({elim['votes']} votes)\n")

                await f.write(f"\n{'=' * 60}\n\n")


def main():
    """Entry point for the game."""
    import sys

    num_players = 6
    model_name = "anthropic/claude-sonnet-4-5-20250929"

    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            num_players = int(sys.argv[1])
            if num_players < 2:
                print("Number of players must be at least 2")
                sys.exit(1)
        except ValueError:
            print("Usage: python main.py [num_players] [model_name]")
            sys.exit(1)

    if len(sys.argv) > 2:
        model_name = sys.argv[2]

    app = SusGame(num_players=num_players, model_name=model_name)
    app.run()


if __name__ == "__main__":
    main()

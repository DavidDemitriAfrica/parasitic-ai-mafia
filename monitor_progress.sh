#!/bin/bash
# Monitor experiment progress

echo "Experiment Progress Monitor"
echo "=========================="
echo ""

# Count total expected games
EXPECTED_GAMES=40

# Count completed games
COMPLETED=$(ls logs/game-*.json 2>/dev/null | wc -l)

# Show progress
echo "Games completed: $COMPLETED / $EXPECTED_GAMES"
echo "Progress: $((COMPLETED * 100 / EXPECTED_GAMES))%"
echo ""

# Show recent logs
echo "Recent completions:"
ls -lt logs/game-*.json 2>/dev/null | head -5 | awk '{print $9}' | xargs -I {} basename {} | sed 's/game-/  /' | sed 's/.json//'

echo ""
echo "Tmux windows status:"
for i in {0..3}; do
    WINDOW="model-$i"
    LAST_LINE=$(tmux capture-pane -t mafia-experiments:$WINDOW -p | tail -1)
    echo "  $WINDOW: $LAST_LINE"
done

echo ""
echo "To attach to session: tmux attach -t mafia-experiments"
echo "To view logs live: tmux select-window -t mafia-experiments:logs"

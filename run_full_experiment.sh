#!/bin/bash
# Run full experimental grid across models with controls
# Each model gets its own tmux window for parallel execution

set -e

MODELS=("openai/gpt-4-turbo" "openai/gpt-4o" "openai/gpt-4o-mini" "openai/gpt-4")
SEED="sive_awakening"  # Single seed for all experiments
NUM_PLAYERS=10
SESSION_NAME="mafia-experiments"

echo "Starting full experimental grid"
echo "Models: ${MODELS[@]}"
echo "Seed: $SEED"
echo "Seeded runs per model: 5"
echo "Control runs per model: 5"
echo "Total: $((${#MODELS[@]} * 10)) games"
echo ""

# Kill existing session if present
tmux kill-session -t $SESSION_NAME 2>/dev/null || true

# Create new session
tmux new-session -d -s $SESSION_NAME -n "monitor"

# Create window for each model
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    WINDOW_NAME="model-$i"

    if [ $i -eq 0 ]; then
        # First window already exists (monitor)
        tmux rename-window -t $SESSION_NAME:0 "$WINDOW_NAME"
    else
        tmux new-window -t $SESSION_NAME -n "$WINDOW_NAME"
    fi

    # Build command to run in this window
    CMD="cd /home/ubuntu/ai-mafia-game && echo 'Running experiments for $MODEL' && "

    # Run 5 seeded experiments
    for j in {1..5}; do
        CMD+="echo '  Seeded run $j/5: $SEED' && "
        CMD+="python run_experiments.py single $SEED --headless --num-players $NUM_PLAYERS --model $MODEL && "
    done

    # Run 5 control experiments
    for j in {1..5}; do
        CMD+="echo '  Control run $j/5' && "
        CMD+="python run_experiments.py single control --headless --num-players $NUM_PLAYERS --model $MODEL && "
    done

    CMD+="echo 'Completed all experiments for $MODEL' && "
    CMD+="echo 'Results saved to logs/' && "
    CMD+="bash"

    # Send command to window
    tmux send-keys -t $SESSION_NAME:$WINDOW_NAME "$CMD" C-m
done

# Create monitoring window
tmux new-window -t $SESSION_NAME -n "logs"
tmux send-keys -t $SESSION_NAME:logs "cd /home/ubuntu/ai-mafia-game && watch -n 5 'ls -lht logs/*.json | head -20'" C-m

# Create analysis window
tmux new-window -t $SESSION_NAME -n "analysis"
tmux send-keys -t $SESSION_NAME:analysis "cd /home/ubuntu/ai-mafia-game && bash" C-m

echo ""
echo "Experiments launched in tmux session: $SESSION_NAME"
echo ""
echo "Commands:"
echo "  tmux attach -t $SESSION_NAME        # Attach to session"
echo "  tmux ls                              # List sessions"
echo "  Ctrl+b w                             # Switch windows (when attached)"
echo "  Ctrl+b d                             # Detach from session"
echo ""
echo "Windows:"
echo "  model-0: ${MODELS[0]}"
echo "  model-1: ${MODELS[1]}"
echo "  model-2: ${MODELS[2]}"
echo "  model-3: ${MODELS[3]}"
echo "  logs: Real-time log monitoring"
echo "  analysis: For running analysis when complete"
echo ""
echo "Estimated completion: 1-2 hours"

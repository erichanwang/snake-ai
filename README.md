# Snake AI - Genetic Algorithm Neural Network

A snake game AI that learns to play using genetic algorithms and neural networks.

## Files

- **main.py** - Headless training mode (AFK friendly) + play mode
- **train.py** - Visual training with live game view, stats, and neural network display
- **snake_game.py** - Snake game logic (grid, movement, collision, scoring)
- **neural_network.py** - Neural network with forward pass, mutation, crossover
- **genetic_algorithm.py** - GA population management, selection, reproduction
- **agent.py** - Snake agent wrapper that uses neural network to decide moves
- **test_game.py** - Manual play mode for testing

## How to Run

**Train AFK (headless, saves automatically):**
```bash
python main.py
```

**Train with visual interface:**
```bash
python train.py
```

**Play manually:**
```bash
python test_game.py
```

**Play with trained AI:**
```bash
python main.py play
```

## How It Works

1. **Neural Network** - 6 inputs (head pos, tail pos, apple pos) → 16 hidden → 4 outputs (directions)
2. **Genetic Algorithm** - 50 agents per generation, top 5 survive, rest are children with mutation
3. **Fitness** - Score * 10000 + survival steps (eating apples is heavily rewarded)
4. **Auto-save** - Best weights saved to `trained_weights.txt` after every generation

Press Ctrl+C to stop training gracefully (saves weights before exit).

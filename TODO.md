# Snake AI Training - Completed

## Changes Made:

### 1. snake_game.py
- Grid size increased from 20x20 to 30x30
- Block size reduced from 20px to 15px for better fit
- Snake now dies when hitting walls (proper collision)
- Snake elongates by 1 when eating apple (no tail pop on eat)
- Added offset parameters to render() for flexible positioning
- Grid lines drawn for better visibility

### 2. train.py (Visual Training)
- Window size increased to 1200x600 for bigger game display
- Game area: 450x450 pixels (30x30 grid * 15px)
- Stats area: 300x450 with fitness graph
- Neural network visualization area: 320x450
- Shows all 6 input nodes, 16 hidden nodes, 4 output nodes
- Connection lines between nodes (sampled for clarity)
- Node labels: head_x/y, tail_x/y, apple_x/y for inputs, up/down/left/right for outputs
- Real-time game visualization of best agent
- Auto-saves after each generation

### 3. main.py
- Updated play_best() for bigger 30x30 grid (500x500 window)
- Loads trained weights before playing
- Centers game in window with 25px padding

### 4. README.md
- Short casual explanation of the project
- File descriptions
- Usage instructions

## How to Use:

**Visual Training (with neural network display):**
```bash
python train.py
```

**Headless Training (AFK):**
```bash
python main.py
```

**Play Trained AI:**
```bash
python main.py play
```

**Manual Play:**
```bash
python test_game.py

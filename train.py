import pygame
import numpy as np
import signal
import sys
import threading
import time
import os
from neural_network import neuralnetwork
from agent import snakeagent
from snake_game import snakegame
from genetic_algorithm import geneticalgorithm

def load_weights(filename="trained_weights.txt"):
    """load weights from file if exists"""
    if not os.path.exists(filename):
        print(f"no existing weights found at {filename}, starting fresh")
        return None
    #create network with 14 inputs and two hidden layers
    brain = neuralnetwork(input_size=14, hidden_size1=16, hidden_size2=16, output_size=4)
    
    with open(filename, "r") as f:
        lines = f.readlines()
    
    #check format by looking for w3 (indicates two hidden layers)
    has_w3 = any("w3:" in line for line in lines)
    
    #count w1 rows to determine input size
    w1_line_count = 0
    for line in lines[1:]:  #skip "w1:" header
        if line.strip() == "b1:":
            break
        w1_line_count += 1
    
    if has_w3 and w1_line_count == 14:
        #new format - 14 inputs, two hidden layers
        #w1: 14 lines
        idx = 1
        w1 = []
        for i in range(14):
            w1.append(list(map(float, lines[idx+i].strip().split(","))))
        brain.w1 = np.array(w1)
        #b1: 1 line
        idx = 16
        brain.b1 = np.array([list(map(float, lines[idx].strip().split(",")))])
        #w2: 16 lines (hidden1 to hidden2)
        idx = 18
        w2 = []
        for i in range(16):
            w2.append(list(map(float, lines[idx+i].strip().split(","))))
        brain.w2 = np.array(w2)
        #b2: 1 line
        idx = 35
        brain.b2 = np.array([list(map(float, lines[idx].strip().split(",")))])
        #w3: 16 lines (hidden2 to output)
        idx = 37
        w3 = []
        for i in range(16):
            w3.append(list(map(float, lines[idx+i].strip().split(","))))
        brain.w3 = np.array(w3)
        #b3: 1 line
        idx = 54
        brain.b3 = np.array([list(map(float, lines[idx].strip().split(",")))])
        print(f"loaded new format weights (14 inputs, 2 hidden layers) from {filename}")
    elif w1_line_count == 13:
        #old format - 13 inputs, one hidden layer (16x4), pad to 14 and restructure
        idx = 1
        w1 = []
        for i in range(13):
            w1.append(list(map(float, lines[idx+i].strip().split(","))))
        #pad 14th input (apple_distance)
        w1.append([0.0] * 16)
        brain.w1 = np.array(w1)
        #b1
        idx = 15
        brain.b1 = np.array([list(map(float, lines[idx].strip().split(",")))])
        #old w2 was 16x4 (hidden to output), this becomes new w3 (hidden2 to output)
        idx = 17
        old_w2 = []
        for i in range(16):
            old_w2.append(list(map(float, lines[idx+i].strip().split(","))))
        #initialize new w2 as identity-like mapping (hidden1 to hidden2)
        brain.w2 = np.eye(16) * 0.1 + np.random.randn(16, 16) * 0.05
        brain.b2 = np.zeros((1, 16))
        #old w2 becomes w3
        brain.w3 = np.array(old_w2)
        #b3 from old b2
        idx = 34
        brain.b3 = np.array([list(map(float, lines[idx].strip().split(",")))])
        print(f"loaded old format weights (13 inputs, restructured to 14->16->16->4) from {filename}")

    else:
        #very old format - just initialize fresh
        print(f"incompatible weight format, starting fresh")
        return None
    
    return brain

class VisualTrainer:
    def __init__(self, pop_size=50):
        self.ga = geneticalgorithm(pop_size=pop_size)
        self.training = True
        self.best_fitness_history = []
        self.lock = threading.Lock()
        #pygame setup
        pygame.init()
        self.screen_width = 1200
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Snake AI - Visual Training")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 12)
        self.big_font = pygame.font.SysFont("monospace", 16)
        #game display areas - bigger for 30x30 grid
        self.game_area = pygame.Rect(50, 50, 450, 450)  #30x30 * 15px = 450px
        self.stats_area = pygame.Rect(520, 50, 300, 450)
        self.nn_area = pygame.Rect(830, 50, 320, 450)  #neural network visualization
        
        #load existing weights if available
        self.load_existing_weights()
        
    def load_existing_weights(self):
        """seed population with existing weights if available"""
        existing_brain = load_weights("trained_weights.txt")
        if existing_brain is not None:
            print("seeding population with existing weights...")
            #replace first agent with loaded weights
            self.ga.population[0] = snakeagent(existing_brain)
            #create variations for next 10 agents
            for i in range(1, min(11, self.ga.pop_size)):
                mutated_brain = existing_brain.copy()
                mutated_brain.mutate(rate=0.05)
                self.ga.population[i] = snakeagent(mutated_brain)
            print(f"seeded {min(11, self.ga.pop_size)} agents with existing weights")
        
    def train_generation(self):
        """Train one generation"""
        self.ga.evolve()
        with self.lock:
            self.best_fitness_history.append(self.ga.best.fitness)
            #keep last 50 gens for display
            if len(self.best_fitness_history) > 50:
                self.best_fitness_history = self.best_fitness_history[-50:]
        #save weights after each gen
        self.ga.save_weights("trained_weights.txt")
        
    def training_loop(self):
        """Background training loop"""
        print("training started in bg...")
        while self.training:
            self.train_generation()
            print(f"gen {self.ga.generation}: best fitness = {self.ga.best.fitness}")
            #small delay to prevent cpu overload
            time.sleep(0.1)
            
    def draw_game(self, game):
        """draw snake game"""
        if game is None:
            #draw empty grid
            pygame.draw.rect(self.screen, (20, 20, 20), self.game_area)
            return
        #draw game with offset
        game.render(self.screen, self.game_area.x, self.game_area.y)
        #draw border
        pygame.draw.rect(self.screen, (100, 100, 100), self.game_area, 2)
        
    def draw_nn(self, brain):
        """draw neural network nodes and connections"""
        if brain is None:
            return
            
        #clear nn area
        pygame.draw.rect(self.screen, (25, 25, 25), self.nn_area)
        
        #nn architecture: 14 inputs -> 16 hidden1 -> 16 hidden2 -> 4 outputs
        input_nodes = 14
        hidden1_nodes = 16
        hidden2_nodes = 16
        output_nodes = 4
        
        #positions - wider spacing for more inputs
        left_x = self.nn_area.x + 40
        hidden1_x = self.nn_area.x + 120
        hidden2_x = self.nn_area.x + 200
        right_x = self.nn_area.x + 280
        start_y = self.nn_area.y + 25
        end_y = self.nn_area.y + self.nn_area.height - 25
        
        #calculate y positions
        input_ys = [start_y + (end_y - start_y) * i / (input_nodes - 1) for i in range(input_nodes)]
        hidden1_ys = [start_y + (end_y - start_y) * i / (hidden1_nodes - 1) for i in range(hidden1_nodes)]
        hidden2_ys = [start_y + (end_y - start_y) * i / (hidden2_nodes - 1) for i in range(hidden2_nodes)]
        output_ys = [start_y + (end_y - start_y) * i / (output_nodes - 1) for i in range(output_nodes)]
        
        #draw connections (sample for visibility)
        #input to hidden1
        for i, iy in enumerate(input_ys):
            for j, hy in enumerate(hidden1_ys):
                if j % 3 == 0:  #sample for clarity
                    weight = abs(brain.w1[i][j]) if i < brain.w1.shape[0] and j < brain.w1.shape[1] else 0
                    #red for negative (danger), green for positive (reward)
                    if i >= 7:  #wall dists, apple dx/dy, apple_distance
                        color = (200, 50, 50) if weight < 0 else (50, 200, 50)
                    else:
                        color = (100, 100, 100) if weight < 0.1 else (150, 150, 150) if weight < 0.5 else (200, 200, 200)
                    pygame.draw.line(self.screen, color, (left_x, iy), (hidden1_x, hy), 1)
        
        #hidden1 to hidden2
        for i, hy in enumerate(hidden1_ys):
            for j, h2y in enumerate(hidden2_ys):
                if i % 4 == 0 and j % 4 == 0:  #sample for clarity
                    weight = abs(brain.w2[i][j]) if i < brain.w2.shape[0] and j < brain.w2.shape[1] else 0
                    color = (100, 100, 100) if weight < 0.1 else (150, 150, 150) if weight < 0.5 else (200, 200, 200)
                    pygame.draw.line(self.screen, color, (hidden1_x, hy), (hidden2_x, h2y), 1)
        
        #hidden2 to output
        for i, h2y in enumerate(hidden2_ys):
            for j, oy in enumerate(output_ys):
                if i % 4 == 0:  #sample for clarity
                    weight = abs(brain.w3[i][j]) if i < brain.w3.shape[0] and j < brain.w3.shape[1] else 0
                    color = (100, 100, 100) if weight < 0.1 else (150, 150, 150) if weight < 0.5 else (200, 200, 200)
                    pygame.draw.line(self.screen, color, (hidden2_x, h2y), (right_x, oy), 1)
        
        #draw nodes
        #input nodes with color coding
        labels = ["head_x", "head_y", "tail_x", "tail_y", "apple_x", "apple_y", "dir",
                  "d_left", "d_right", "d_top", "d_bot", "to_ap_x", "to_ap_y", "ap_dist"]
        
        for i, (y, label) in enumerate(zip(input_ys, labels)):
            #color code: blue for position, red for wall dist (danger), green for apple (reward)
            if i < 6:
                color = (0, 150, 255)  #position inputs
            elif i == 6:
                color = (255, 200, 0)  #direction
            elif i < 11:
                color = (255, 100, 100)  #wall distances (danger - red tint)
            elif i < 13:
                color = (100, 255, 100)  #apple direction (reward - green tint)
            else:
                color = (255, 255, 100)  #apple distance (yellow)
            pygame.draw.circle(self.screen, color, (left_x, int(y)), 5)
            text = self.font.render(label, True, (200, 200, 200))
            self.screen.blit(text, (left_x - 40, int(y) - 4))
        
        #hidden1 nodes
        for i, y in enumerate(hidden1_ys):
            pygame.draw.circle(self.screen, (255, 150, 0), (hidden1_x, int(y)), 4)
        
        #hidden2 nodes
        for i, y in enumerate(hidden2_ys):
            pygame.draw.circle(self.screen, (255, 100, 150), (hidden2_x, int(y)), 4)
        
        #output nodes
        actions = ["up", "down", "left", "right"]
        for i, (y, label) in enumerate(zip(output_ys, actions)):
            pygame.draw.circle(self.screen, (0, 255, 100), (right_x, int(y)), 7)
            text = self.font.render(label, True, (200, 200, 200))
            self.screen.blit(text, (right_x + 10, int(y) - 5))
        
        #title
        title = self.big_font.render("NN (14->16->16->4)", True, (255, 255, 255))
        self.screen.blit(title, (self.nn_area.x + 50, self.nn_area.y + 5))
        
        #legend
        legend_text = self.font.render("Red=danger, Green=reward", True, (150, 150, 150))
        self.screen.blit(legend_text, (self.nn_area.x + 40, self.nn_area.y + 420))
        
        #border
        pygame.draw.rect(self.screen, (100, 100, 100), self.nn_area, 2)
        
    def draw_stats(self):
        """draw stats"""
        #clear stats area
        pygame.draw.rect(self.screen, (30, 30, 30), self.stats_area)
        #title
        title = self.big_font.render("Training Statistics", True, (255, 255, 255))
        self.screen.blit(title, (self.stats_area.x + 50, self.stats_area.y + 10))
        with self.lock:
            gen_text = self.font.render(f"Generation: {self.ga.generation}", True, (200, 200, 200))
            self.screen.blit(gen_text, (self.stats_area.x + 20, self.stats_area.y + 50))
            if hasattr(self.ga, 'best') and self.ga.best:
                fitness_text = self.font.render(f"Best Fitness: {self.ga.best.fitness:.0f}", True, (200, 200, 200))
                self.screen.blit(fitness_text, (self.stats_area.x + 20, self.stats_area.y + 75))
                score_text = self.font.render(f"Best Score: {self.ga.best.fitness // 10000}", True, (200, 200, 200))
                self.screen.blit(score_text, (self.stats_area.x + 20, self.stats_area.y + 100))
                snake_len = self.font.render(f"Snake Length: {self.ga.best.fitness // 10000 + 1}", True, (200, 200, 200))
                self.screen.blit(snake_len, (self.stats_area.x + 20, self.stats_area.y + 125))
            #draw fitness graph
            if len(self.best_fitness_history) > 1:
                graph_rect = pygame.Rect(self.stats_area.x + 20, self.stats_area.y + 160, 260, 120)
                pygame.draw.rect(self.screen, (50, 50, 50), graph_rect)
                max_fitness = max(self.best_fitness_history) if self.best_fitness_history else 1
                min_fitness = min(self.best_fitness_history) if self.best_fitness_history else 0
                fitness_range = max_fitness - min_fitness if max_fitness != min_fitness else 1
                points = []
                for i, fitness in enumerate(self.best_fitness_history):
                    x = graph_rect.x + (i / (len(self.best_fitness_history) - 1)) * graph_rect.width
                    y = graph_rect.y + graph_rect.height - ((fitness - min_fitness) / fitness_range) * graph_rect.height
                    points.append((x, y))
                if len(points) > 1:
                    pygame.draw.lines(self.screen, (0, 200, 0), False, points, 2)
        #instructions
        instr_text = self.font.render("Press 'Q' to quit, 'S' to save", True, (150, 150, 150))
        self.screen.blit(instr_text, (self.stats_area.x + 20, self.stats_area.y + 300))
        instr2_text = self.font.render("Red=Apple, Green=Snake", True, (150, 150, 150))
        self.screen.blit(instr2_text, (self.stats_area.x + 20, self.stats_area.y + 320))
        #border
        pygame.draw.rect(self.screen, (100, 100, 100), self.stats_area, 2)
        
    def run(self):
        """main loop"""
        #start bg training thread
        train_thread = threading.Thread(target=self.training_loop)
        train_thread.daemon = True
        train_thread.start()
        #signal handler
        def signal_handler(sig, frame):
            print("\nsaving weights...")
            self.ga.save_weights("trained_weights.txt")
            self.training = False
            pygame.quit()
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)
        #game viz
        demo_game = None
        demo_agent = None
        demo_steps = 0
        max_demo_steps = 200
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_s:
                        self.ga.save_weights("trained_weights.txt")
                        print("weights saved manually")
            #clear screen
            self.screen.fill((10, 10, 10))
            #update demo game w/best agent
            with self.lock:
                if hasattr(self.ga, 'best') and self.ga.best:
                    if demo_game is None or not demo_agent or demo_steps >= max_demo_steps:
                        demo_game = snakegame()
                        demo_state = demo_game.reset()
                        demo_agent = snakeagent(self.ga.best.brain.copy())
                        demo_steps = 0
                    action = demo_agent.get_action(demo_state)
                    demo_state, alive, score = demo_game.step(action)
                    demo_steps += 1
                    if not alive:
                        demo_game = None
                        demo_agent = None
                    self.draw_game(demo_game)
                    self.draw_nn(self.ga.best.brain)
            self.draw_stats()
            pygame.display.flip()
            self.clock.tick(15)
        #cleanup
        self.training = False
        self.ga.save_weights("trained_weights.txt")
        print("saved.")
        pygame.quit()

def main():
    print("=" * 50)
    print("snake ai visual trainer")
    print("=" * 50)
    print("continuous training w/auto-save")
    print("press 'Q' to quit, 'S' to save")
    print("=" * 50)
    trainer = VisualTrainer(pop_size=50)
    trainer.run()

if __name__ == "__main__":
    main()

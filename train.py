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
    brain=neuralnetwork(input_size=7)  #7 inputs now
    with open(filename,"r") as f:
        lines=f.readlines()
    #check if old format (6 inputs) or new format (7 inputs)
    #new format has 7 rows for w1, old has 6
    is_new_format = len(lines) > 28  #new has more lines
    if is_new_format:
        #parse w1:7lines (new format)
        idx=1
        w1=[]
        for i in range(7):
            w1.append(list(map(float,lines[idx+i].strip().split(","))))
        brain.w1=np.array(w1)
        #parse b1:1line
        idx=9
        brain.b1=np.array([list(map(float,lines[idx].strip().split(",")))])
        #parse w2:16lines
        idx=11
        w2=[]
        for i in range(16):
            w2.append(list(map(float,lines[idx+i].strip().split(","))))
        brain.w2=np.array(w2)
        #parse b2:1line
        idx=28
        brain.b2=np.array([list(map(float,lines[idx].strip().split(",")))])
        print(f"loaded new format weights (7 inputs) from {filename}")
    else:
        #old format - 6 inputs, need to pad to 7
        idx=1
        w1=[]
        for i in range(6):
            w1.append(list(map(float,lines[idx+i].strip().split(","))))
        #pad with zeros for 7th input
        w1.append([0.0]*16)
        brain.w1=np.array(w1)
        #parse b1:1line
        idx=8
        brain.b1=np.array([list(map(float,lines[idx].strip().split(",")))])
        #parse w2:16lines
        idx=10
        w2=[]
        for i in range(16):
            w2.append(list(map(float,lines[idx+i].strip().split(","))))
        brain.w2=np.array(w2)
        #parse b2:1line
        idx=27
        brain.b2=np.array([list(map(float,lines[idx].strip().split(",")))])
        print(f"loaded old format weights (6 inputs, padded to 7) from {filename}")
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
        
        #nn architecture: 7 inputs -> 16 hidden -> 4 outputs
        input_nodes = 7

        hidden_nodes = 16
        output_nodes = 4
        
        #positions
        left_x = self.nn_area.x + 40
        hidden_x = self.nn_area.x + 140
        right_x = self.nn_area.x + 240
        start_y = self.nn_area.y + 40
        end_y = self.nn_area.y + self.nn_area.height - 40
        
        #calculate y positions
        input_ys = [start_y + (end_y - start_y) * i / (input_nodes - 1) for i in range(input_nodes)]
        hidden_ys = [start_y + (end_y - start_y) * i / (hidden_nodes - 1) for i in range(hidden_nodes)]
        output_ys = [start_y + (end_y - start_y) * i / (output_nodes - 1) for i in range(output_nodes)]
        
        #draw connections (sample for visibility)
        #input to hidden
        for i, iy in enumerate(input_ys):
            for j, hy in enumerate(hidden_ys):
                if j % 2 == 0:  #sample every other for clarity
                    weight = abs(brain.w1[i][j]) if i < brain.w1.shape[0] and j < brain.w1.shape[1] else 0
                    color = (100, 100, 100) if weight < 0.1 else (150, 150, 150) if weight < 0.5 else (200, 200, 200)
                    pygame.draw.line(self.screen, color, (left_x, iy), (hidden_x, hy), 1)
        
        #hidden to output
        for i, hy in enumerate(hidden_ys):
            for j, oy in enumerate(output_ys):
                if i % 4 == 0:  #sample for clarity
                    weight = abs(brain.w2[i][j]) if i < brain.w2.shape[0] and j < brain.w2.shape[1] else 0
                    color = (100, 100, 100) if weight < 0.1 else (150, 150, 150) if weight < 0.5 else (200, 200, 200)
                    pygame.draw.line(self.screen, color, (hidden_x, hy), (right_x, oy), 1)
        
        #draw nodes
        #input nodes
        labels = ["head_x", "head_y", "tail_x", "tail_y", "apple_x", "apple_y", "direction"]

        for i, (y, label) in enumerate(zip(input_ys, labels)):
            pygame.draw.circle(self.screen, (0, 150, 255), (left_x, int(y)), 8)
            text = self.font.render(label, True, (200, 200, 200))
            self.screen.blit(text, (left_x - 35, int(y) - 6))
        
        #hidden nodes
        for i, y in enumerate(hidden_ys):
            pygame.draw.circle(self.screen, (255, 150, 0), (hidden_x, int(y)), 6)
        
        #output nodes
        actions = ["up", "down", "left", "right"]
        for i, (y, label) in enumerate(zip(output_ys, actions)):
            pygame.draw.circle(self.screen, (0, 255, 100), (right_x, int(y)), 8)
            text = self.font.render(label, True, (200, 200, 200))
            self.screen.blit(text, (right_x + 12, int(y) - 6))
        
        #title
        title = self.big_font.render("Neural Network", True, (255, 255, 255))
        self.screen.blit(title, (self.nn_area.x + 80, self.nn_area.y + 10))
        
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

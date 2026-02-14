import pygame
import numpy as np
import signal
import sys
import threading
import time
from neural_network import neuralnetwork
from agent import snakeagent
from snake_game import snakegame
from genetic_algorithm import geneticalgorithm

class VisualTrainer:
    def __init__(self, pop_size=50):
        self.ga = geneticalgorithm(pop_size=pop_size)
        self.training = True
        self.best_fitness_history = []
        self.lock = threading.Lock()
        
        #pygame setup
        pygame.init()

        self.screen = pygame.display.set_mode((800, 400))
        pygame.display.set_caption("Snake AI - Visual Training")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 14)
        self.big_font = pygame.font.SysFont("monospace", 20)
        
        #game display areas
        self.game_area = pygame.Rect(50, 50, 300, 300)
        self.stats_area = pygame.Rect(400, 50, 350, 300)

        
    def train_generation(self):
        """Train one generation"""
        self.ga.evolve()
        with self.lock:
            self.best_fitness_history.append(self.ga.best.fitness)
            # Keep only last 50 generations for display
            if len(self.best_fitness_history) > 50:
                self.best_fitness_history = self.best_fitness_history[-50:]
        # Save weights after each generation
        self.ga.save_weights("trained_weights.txt")
        
    def training_loop(self):
        """Background training loop"""
        print("Training started in background...")
        while self.training:
            self.train_generation()
            print(f"Gen {self.ga.generation}: best fitness = {self.ga.best.fitness}")
            # Small delay to prevent CPU overload
            time.sleep(0.1)
            
    def draw_game(self, game):
        """draw snake game"""
        #clear game area

        pygame.draw.rect(self.screen, (20, 20, 20), self.game_area)
        
        cell_size = 30  #10x10 grid
        offset_x = self.game_area.x
        offset_y = self.game_area.y
        
        #draw grid

        for i in range(11):
            pygame.draw.line(self.screen, (40, 40, 40), 
                           (offset_x + i * cell_size, offset_y),
                           (offset_x + i * cell_size, offset_y + 300))
            pygame.draw.line(self.screen, (40, 40, 40),
                           (offset_x, offset_y + i * cell_size),
                           (offset_x + 300, offset_y + i * cell_size))
        
        #draw snake
        for segment in game.snake:

            x, y = segment
            rect = pygame.Rect(offset_x + x * cell_size + 1, 
                             offset_y + y * cell_size + 1, 
                             cell_size - 2, cell_size - 2)
            pygame.draw.rect(self.screen, (0, 200, 0), rect)
            
        #draw apple
        ax, ay = game.apple

        rect = pygame.Rect(offset_x + ax * cell_size + 1,
                         offset_y + ay * cell_size + 1,
                         cell_size - 2, cell_size - 2)
        pygame.draw.rect(self.screen, (200, 0, 0), rect)
        
    def draw_stats(self):
        """draw stats"""
        #clear stats area

        pygame.draw.rect(self.screen, (30, 30, 30), self.stats_area)
        
        #title
        title = self.big_font.render("Training Statistics", True, (255, 255, 255))

        self.screen.blit(title, (self.stats_area.x + 10, self.stats_area.y + 10))
        
        with self.lock:
            gen_text = self.font.render(f"Generation: {self.ga.generation}", True, (200, 200, 200))
            self.screen.blit(gen_text, (self.stats_area.x + 10, self.stats_area.y + 50))
            
            if hasattr(self.ga, 'best') and self.ga.best:
                fitness_text = self.font.render(f"Best Fitness: {self.ga.best.fitness:.0f}", True, (200, 200, 200))
                self.screen.blit(fitness_text, (self.stats_area.x + 10, self.stats_area.y + 75))
                
                score_text = self.font.render(f"Best Score: {self.ga.best.fitness // 10000}", True, (200, 200, 200))
                self.screen.blit(score_text, (self.stats_area.x + 10, self.stats_area.y + 100))
            
            #draw fitness graph
            if len(self.best_fitness_history) > 1:

                graph_rect = pygame.Rect(self.stats_area.x + 10, self.stats_area.y + 140, 330, 100)
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
        instr_text = self.font.render("Press 'Q' to quit, 'S' to save weights", True, (150, 150, 150))

        self.screen.blit(instr_text, (self.stats_area.x + 10, self.stats_area.y + 260))
        
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
        max_demo_steps = 100
        
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
                        print("Weights saved manually")
                        
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
            
            self.draw_stats()
            
            pygame.display.flip()
            self.clock.tick(10)
            
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

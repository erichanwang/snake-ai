import pygame
from neural_network import neuralnetwork
from agent import snakeagent
from snake_game import snakgame
from genetic_algorithm import geneticalgorithm

def train():
    ga=geneticalgorithm(pop_size=50)
    for gen in range(100):
        ga.evolve()
        best=ga.get_best()
        print(f"gen {gen}: best fitness={best.fitness}")

def play_best():
    pygame.init()
    screen=pygame.display.set_mode((400,400))
    pygame.display.set_caption("ai snake")
    
    brain=neuralnetwork()
    agent=snakeagent(brain)
    game=snakgame()
    state=game.reset()
    
    clock=pygame.time.Clock()
    running=True
    
    while running:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running=False
        
        action=agent.get_action(state)
        state,alive,score=game.step(action)
        game.render(screen)
        pygame.display.flip()
        clock.tick(10)
        
        if not alive:
            print(f"game over! score: {score}")
            state=game.reset()
    
    pygame.quit()

if __name__=="__main__":
    play_best()

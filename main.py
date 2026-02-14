import pygame
from neural_network import neuralnetwork
from agent import snakeagent
from snake_game import snakegame

from genetic_algorithm import geneticalgorithm

def save_weights(brain,filename="weights.txt"):
    with open(filename,"w") as f:
        f.write("w1:\n")
        for row in brain.w1:
            f.write(",".join(map(str,row))+"\n")
        f.write("b1:\n")
        f.write(",".join(map(str,brain.b1[0]))+"\n")
        f.write("w2:\n")
        for row in brain.w2:
            f.write(",".join(map(str,row))+"\n")
        f.write("b2:\n")
        f.write(",".join(map(str,brain.b2[0]))+"\n")
    print(f"weights saved to {filename}")

def load_weights(filename="trained_weights.txt"):
    brain=neuralnetwork()
    with open(filename,"r") as f:
        lines=f.readlines()
    #parse w1 (6 lines after "w1:")
    idx=1
    w1=[]
    for i in range(6):
        w1.append(list(map(float,lines[idx+i].strip().split(","))))
    brain.w1=np.array(w1)
    #parse b1 (1 line after "b1:")
    idx=8
    brain.b1=np.array([list(map(float,lines[idx].strip().split(",")))])
    #parse w2 (16 lines after "w2:")
    idx=10
    w2=[]
    for i in range(16):
        w2.append(list(map(float,lines[idx+i].strip().split(","))))
    brain.w2=np.array(w2)
    #parse b2 (1 line after "b2:")
    idx=27
    brain.b2=np.array([list(map(float,lines[idx].strip().split(",")))])
    print(f"weights loaded from {filename}")
    return brain


def train():
    ga=geneticalgorithm(pop_size=50)
    for gen in range(100):
        ga.evolve()
        print(f"gen {gen}: best fitness={ga.best.fitness}")
        save_weights(ga.best.brain,"trained_weights.txt")




def play_best():
    pygame.init()
    screen=pygame.display.set_mode((400,400))
    pygame.display.set_caption("ai snake")
    
    brain=neuralnetwork()
    agent=snakeagent(brain)
    game=snakegame()

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
    import sys
    if len(sys.argv)>1 and sys.argv[1]=="play":
        play_best()
    else:
        train()

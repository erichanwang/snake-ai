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
    #parse w1:6lines
    idx=1
    w1=[]
    for i in range(6):
        w1.append(list(map(float,lines[idx+i].strip().split(","))))
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
    print(f"weights loaded from {filename}")
    return brain

def train():
    import signal
    import sys
    ga=geneticalgorithm(pop_size=50)
    def signal_handler(sig,frame):
        print("\nInterrupt received! Saving weights...")
        ga.save_weights("trained_weights.txt")
        sys.exit(0)
    signal.signal(signal.SIGINT,signal_handler)
    signal.signal(signal.SIGTERM,signal_handler)
    #start continuous training
    print("training... (ctrl+c to stop)")
    print("auto-save enabled")
    try:
        while True:
            ga.evolve()
            print(f"gen {ga.generation}: best fitness={ga.best.fitness}")
            ga.save_weights("trained_weights.txt")
    except KeyboardInterrupt:
        #user stopped training
        print("\nstopped.")
        ga.save_weights("trained_weights.txt")
        print("saved.")

def play_best():
    pygame.init()
    #bigger window for 30x30 grid (450x450) + some padding
    screen=pygame.display.set_mode((500,500))
    pygame.display.set_caption("ai snake")
    brain=load_weights("trained_weights.txt")
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
        #center the game in the window
        screen.fill((0,0,0))
        game.render(screen,25,25)
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

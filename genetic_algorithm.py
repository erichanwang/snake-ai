import random
from neural_network import neuralnetwork
from agent import snakeagent
from snake_game import snakgame

class geneticalgorithm:
    def __init__(self,pop_size=50):
        self.pop_size=pop_size
        self.population=[]
        for _ in range(pop_size):
            brain=neuralnetwork()
            self.population.append(snakeagent(brain))
    
    def evaluate(self):
        for agent in self.population:
            game=snakgame()
            state=game.reset()
            while True:
                action=agent.get_action(state)
                state,alive,score=game.step(action)
                if not alive:
                    break
            agent.fitness=score
    
    def selection(self):
        self.population.sort(key=lambda a:a.fitness,reverse=True)
        return self.population[:self.pop_size//2]
    
    def reproduce(self,parents):
        new_pop=[]
        while len(new_pop)<self.pop_size:
            p1=random.choice(parents)
            p2=random.choice(parents)
            child_brain=neuralnetwork.crossover(p1.brain,p2.brain)
            child_brain.mutate()
            new_pop.append(snakeagent(child_brain))
        self.population=new_pop
    
    def evolve(self):
        self.evaluate()
        parents=self.selection()
        self.reproduce(parents)
    
    def get_best(self):
        return max(self.population,key=lambda a:a.fitness)

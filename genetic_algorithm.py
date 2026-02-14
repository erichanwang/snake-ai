import random
import numpy as np
from neural_network import neuralnetwork
from agent import snakeagent
from snake_game import snakegame


class geneticalgorithm:
    def __init__(self,pop_size=100):
        self.pop_size=pop_size
        self.population=[]
        self.generation=0
        for _ in range(pop_size):
            brain=neuralnetwork()
            self.population.append(snakeagent(brain))

    
    def evaluate(self):
        for agent in self.population:
            game=snakegame()
            state=game.reset()
            steps=0
            max_steps=1000  #limit steps to prevent infinite loops
            while steps<max_steps:
                action=agent.get_action(state)
                state,alive,score=game.step(action)
                steps+=1
                if not alive:
                    break
            #fitness=score*10000+survival bonus
            agent.fitness=score*10000+steps



    
    def selection(self):
        self.population.sort(key=lambda a:a.fitness,reverse=True)
        return self.population[:self.pop_size//2]
    
    def reproduce(self,parents):
        new_pop=[]
        #elitism:keep top5
        for p in parents[:5]:
            new_pop.append(snakeagent(p.brain.copy()))
        #fill w/children
        while len(new_pop)<self.pop_size:
            p1=random.choice(parents)
            p2=random.choice(parents)
            child_brain=neuralnetwork.crossover(p1.brain,p2.brain)
            child_brain.mutate(rate=0.02)
            new_pop.append(snakeagent(child_brain))
        self.population=new_pop


    
    def evolve(self):
        self.evaluate()
        self.best=self.get_best()
        parents=self.selection()
        self.reproduce(parents)
        self.generation+=1
    
    def get_best(self):
        return max(self.population,key=lambda a:a.fitness)
    
    def save_weights(self,filename="trained_weights.txt"):
        """save best weights to file"""
        if not hasattr(self,'best') or self.best is None:
            return

        brain=self.best.brain
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
            f.write("w3:\n")
            for row in brain.w3:
                f.write(",".join(map(str,row))+"\n")
            f.write("b3:\n")
            f.write(",".join(map(str,brain.b3[0]))+"\n")
        print(f"Generation {self.generation}: weights saved to {filename} (fitness={self.best.fitness})")

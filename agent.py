import random

class snakeagent:
    def __init__(self,brain=None):
        self.brain=brain
        self.fitness=0
    
    def get_action(self,state):
        if self.brain:
            return self.brain.get_action(state)
        return random.randint(0,3)
    
    def reset(self):
        self.fitness=0

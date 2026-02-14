import pygame
import random

class snakegame:

    def __init__(self,width=20,height=20):
        self.width=width
        self.height=height
        self.block_size=20
        self.reset()
    
    def reset(self):
        #init snake at center
        self.snake=[(self.width//2,self.height//2)]
        self.direction=(0,-1)#up
        self.score=0
        self.steps=0
        self.max_steps=2000
        self.place_apple()
        return self.get_state()
    
    def place_apple(self):
        while True:
            self.apple=(random.randint(0,self.width-1),random.randint(0,self.height-1))
            if self.apple not in self.snake:
                break
    
    def get_state(self):
        #input: head_x, head_y, tail_x, tail_y, apple_x, apple_y
        head=self.snake[0]
        tail=self.snake[-1] if len(self.snake)>1 else head
        return (
            head[0]/self.width,
            head[1]/self.height,
            tail[0]/self.width,
            tail[1]/self.height,
            self.apple[0]/self.width,
            self.apple[1]/self.height
        )
    
    def step(self,action):
        #action:0=up,1=down,2=left,3=right
        directions=[(0,-1),(0,1),(-1,0),(1,0)]
        self.direction=directions[action]
        
        head=self.snake[0]
        new_head=(head[0]+self.direction[0],head[1]+self.direction[1])
        
        #wall collision
        if new_head[0]<0 or new_head[0]>=self.width or new_head[1]<0 or new_head[1]>=self.height:
            return self.get_state(),False,self.score
        
        #self collision
        if new_head in self.snake:
            return self.get_state(),False,self.score
        
        self.snake.insert(0,new_head)
        
        #eat apple
        if new_head==self.apple:
            self.score+=1
            self.place_apple()
        else:
            self.snake.pop()
        
        self.steps+=1
        if self.steps>=self.max_steps:
            return self.get_state(),False,self.score
        
        return self.get_state(),True,self.score
    
    def is_game_over(self):
        head=self.snake[0]
        if head[0]<0 or head[0]>=self.width or head[1]<0 or head[1]>=self.height:
            return True
        if head in self.snake[1:]:
            return True
        return False
    
    def render(self,screen):
        screen.fill((0,0,0))
        #draw snake
        for i,segment in enumerate(self.snake):
            color=(0,255,0) if i==0 else (0,200,0)
            pygame.draw.rect(screen,color,(segment[0]*self.block_size,segment[1]*self.block_size,self.block_size-2,self.block_size-2))
        #draw apple
        pygame.draw.rect(screen,(255,0,0),(self.apple[0]*self.block_size,self.apple[1]*self.block_size,self.block_size-2,self.block_size-2))
        #draw score
        font=pygame.font.Font(None,36)
        text=font.render(f"score:{self.score}",True,(255,255,255))
        screen.blit(text,(10,10))

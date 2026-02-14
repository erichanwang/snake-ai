import pygame
import random
import time
import math

class snakegame:

    def __init__(self,width=30,height=30):
        self.width=width
        self.height=height
        self.block_size=15
        self.reset()

    
    def reset(self):
        #init snake at center
        self.snake=[(self.width//2,self.height//2)]
        self.direction=(0,-1)#up
        self.score=0
        self.steps=0
        self.max_steps=2000
        #track position visits to detect looping
        self.position_history={}  #pos->list of timestamps
        self.loop_threshold=5  #visits in 3 sec
        self.loop_window=3.0  #seconds
        #track apple eating for stagnation detection
        self.last_apple_time=time.time()
        self.apple_timeout=10.0  #seconds to get next apple
        self.place_apple()

        return self.get_state()

    
    def place_apple(self):
        while True:
            self.apple=(random.randint(0,self.width-1),random.randint(0,self.height-1))
            if self.apple not in self.snake:
                break
    
    def get_direction_code(self):
        """return direction as 0=up,1=down,2=left,3=right"""
        if self.direction==(0,-1):
            return 0
        elif self.direction==(0,1):
            return 1
        elif self.direction==(-1,0):
            return 2
        elif self.direction==(1,0):
            return 3
        return 0
    
    def get_state(self):
        #input: head_x, head_y, tail_x, tail_y, apple_x, apple_y, direction,
        #        dist_left, dist_right, dist_top, dist_bottom, apple_dx, apple_dy, apple_distance
        head=self.snake[0]
        tail=self.snake[-1] if len(self.snake)>1 else head
        direction_code=self.get_direction_code()
        
        #distance to walls (negative when close - danger signal)
        dist_left=-head[0]  #0 at left wall, more negative further right
        dist_right=-(self.width-1-head[0])  #0 at right wall
        dist_top=-head[1]  #0 at top wall
        dist_bottom=-(self.height-1-head[1])  #0 at bottom wall
        
        #signed distance to apple (positive = apple is in that direction)
        apple_dx=self.apple[0]-head[0]  #positive if apple to the right
        apple_dy=self.apple[1]-head[1]  #positive if apple below
        
        #Euclidean distance to apple (normalized)
        apple_distance=math.sqrt(apple_dx**2+apple_dy**2)
        max_dist=math.sqrt(self.width**2+self.height**2)
        
        return (
            head[0]/self.width,
            head[1]/self.height,
            tail[0]/self.width,
            tail[1]/self.height,
            self.apple[0]/self.width,
            self.apple[1]/self.height,
            direction_code/3.0,  #normalize to 0-1
            dist_left/self.width,  #negative near left wall
            dist_right/self.width,  #negative near right wall
            dist_top/self.height,  #negative near top wall
            dist_bottom/self.height,  #negative near bottom wall
            apple_dx/self.width,  #positive if apple right
            apple_dy/self.height,  #positive if apple below
            apple_distance/max_dist  #normalized Euclidean distance (0-1, smaller is closer)
        )

    
    def check_looping(self,pos):
        """check if snake is stuck visiting same positions"""
        current_time=time.time()
        if pos not in self.position_history:
            self.position_history[pos]=[]
        #add current visit
        self.position_history[pos].append(current_time)
        #remove old visits outside window
        self.position_history[pos]=[t for t in self.position_history[pos] if current_time-t<=self.loop_window]
        #check if visited too many times
        return len(self.position_history[pos])>self.loop_threshold

    def step(self,action):
        #action:0=up,1=down,2=left,3=right
        directions=[(0,-1),(0,1),(-1,0),(1,0)]
        new_direction=directions[action]
        
        #prevent 180-degree turns (can't reverse direction)
        #if going up, can't go down; if going left, can't go right
        if (self.direction[0]==-new_direction[0] and self.direction[1]==-new_direction[1]):
            #invalid turn - keep current direction
            new_direction=self.direction
        
        self.direction=new_direction
        
        head=self.snake[0]
        new_head=(head[0]+self.direction[0],head[1]+self.direction[1])
        
        #die on wall collision
        if new_head[0]<0 or new_head[0]>=self.width or new_head[1]<0 or new_head[1]>=self.height:
            return self.get_state(),False,self.score
        
        #die on self collision (fatal)
        if new_head in self.snake:
            return self.get_state(),False,self.score
        
        #check if snake is stuck looping
        self.check_looping(new_head)
        
        self.snake.insert(0,new_head)
        
        #eat apple - elongate by 1 (don't pop tail)
        if new_head==self.apple:
            self.score+=1
            self.last_apple_time=time.time()  #reset timer
            self.place_apple()
            #snake grows by 1, no pop()
        else:
            self.snake.pop()
        
        self.steps+=1
        
        return self.get_state(),True,self.score

    
    def is_game_over(self):
        head=self.snake[0]
        if head[0]<0 or head[0]>=self.width or head[1]<0 or head[1]>=self.height:
            return True
        #also check self collision
        if head in self.snake[1:]:
            return True
        return False
    
    def render(self,screen,offset_x=0,offset_y=0):
        #draw grid background
        for x in range(self.width+1):
            pygame.draw.line(screen,(40,40,40),
                           (offset_x+x*self.block_size,offset_y),
                           (offset_x+x*self.block_size,offset_y+self.height*self.block_size))
        for y in range(self.height+1):
            pygame.draw.line(screen,(40,40,40),
                           (offset_x,offset_y+y*self.block_size),
                           (offset_x+self.width*self.block_size,offset_y+y*self.block_size))
        #draw snake
        for i,segment in enumerate(self.snake):
            color=(0,255,0) if i==0 else (0,200,0)
            pygame.draw.rect(screen,color,(offset_x+segment[0]*self.block_size,offset_y+segment[1]*self.block_size,self.block_size-2,self.block_size-2))
        #draw apple
        pygame.draw.rect(screen,(255,0,0),(offset_x+self.apple[0]*self.block_size,offset_y+self.apple[1]*self.block_size,self.block_size-2,self.block_size-2))
        #draw score
        font=pygame.font.Font(None,36)
        text=font.render(f"score:{self.score}",True,(255,255,255))
        screen.blit(text,(offset_x+10,offset_y+10))

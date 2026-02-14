import pygame
import random
import time

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
        #input: head_x, head_y, tail_x, tail_y, apple_x, apple_y, direction
        head=self.snake[0]
        tail=self.snake[-1] if len(self.snake)>1 else head
        direction_code=self.get_direction_code()
        return (
            head[0]/self.width,
            head[1]/self.height,
            tail[0]/self.width,
            tail[1]/self.height,
            self.apple[0]/self.width,
            self.apple[1]/self.height,
            direction_code/3.0  #normalize to 0-1
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

    def check_edge_case(self,action):
        """manually handle edge cases - force turn at corners only"""
        head=self.snake[0]
        x,y=head[0],head[1]
        direction=self.get_direction_code()
        
        #only handle corners - let snake hit walls normally
        #top right corner going up -> turn left
        if x>=self.width-2 and y<=1 and direction==0:
            return 2  #force left
        #top right corner going right -> turn down
        if x>=self.width-2 and y<=1 and direction==3:
            return 1  #force down
        #top left corner going up -> turn right
        if x<=1 and y<=1 and direction==0:
            return 3  #force right
        #top left corner going left -> turn down
        if x<=1 and y<=1 and direction==2:
            return 1  #force down
        #bottom right corner going down -> turn left
        if x>=self.width-2 and y>=self.height-2 and direction==1:
            return 2  #force left
        #bottom right corner going right -> turn up
        if x>=self.width-2 and y>=self.height-2 and direction==3:
            return 0  #force up
        #bottom left corner going down -> turn right
        if x<=1 and y>=self.height-2 and direction==1:
            return 3  #force right
        #bottom left corner going left -> turn up
        if x<=1 and y>=self.height-2 and direction==2:
            return 0  #force up
        
        return action  #no edge case, use nn action


    def step(self,action):
        #apply edge case handling
        action=self.check_edge_case(action)
        
        #action:0=up,1=down,2=left,3=right
        directions=[(0,-1),(0,1),(-1,0),(1,0)]
        self.direction=directions[action]
        
        head=self.snake[0]
        new_head=(head[0]+self.direction[0],head[1]+self.direction[1])
        
        #only die on wall collision
        if new_head[0]<0 or new_head[0]>=self.width or new_head[1]<0 or new_head[1]>=self.height:
            return self.get_state(),False,self.score
        
        #self collision - don't die, just don't move there
        if new_head in self.snake:
            #stay in place, don't die
            return self.get_state(),True,self.score
        
        #check if snake is stuck looping - don't die, just track
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

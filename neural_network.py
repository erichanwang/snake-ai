import numpy as np

class neuralnetwork:
    def __init__(self, input_size=14, hidden_size1=16, hidden_size2=16, output_size=4):
        #input: head_x, head_y, tail_x, tail_y, apple_x, apple_y, direction,
        #        dist_left, dist_right, dist_top, dist_bottom, apple_dx, apple_dy, apple_distance
        #dist to walls: negative near wall (danger), positive away (safe)
        #apple_dx/dy: positive if apple in that direction (reward)
        #apple_distance: normalized Euclidean distance to apple
        #output: up, down, left, right
        
        self.input_size=input_size
        self.hidden_size1=hidden_size1
        self.hidden_size2=hidden_size2
        self.output_size=output_size
        
        #weights - two hidden layers for more learning capacity
        self.w1=np.random.randn(input_size,hidden_size1)*0.1
        self.b1=np.zeros((1,hidden_size1))
        self.w2=np.random.randn(hidden_size1,hidden_size2)*0.1
        self.b2=np.zeros((1,hidden_size2))
        self.w3=np.random.randn(hidden_size2,output_size)*0.1
        self.b3=np.zeros((1,output_size))


    
    def sigmoid(self,x):
        return 1/(1+np.exp(-np.clip(x,-500,500)))
    
    def relu(self,x):
        return np.maximum(0,x)
    
    def softmax(self,x):
        exp_x=np.exp(x-np.max(x,axis=1,keepdims=True))
        return exp_x/np.sum(exp_x,axis=1,keepdims=True)
    
    def forward(self,x):
        x=np.array(x).reshape(1,-1)
        #input to hidden1
        self.z1=np.dot(x,self.w1)+self.b1
        self.a1=self.relu(self.z1)
        #hidden1 to hidden2
        self.z2=np.dot(self.a1,self.w2)+self.b2
        self.a2=self.relu(self.z2)
        #hidden2 to output
        self.z3=np.dot(self.a2,self.w3)+self.b3
        self.output=self.softmax(self.z3)
        return self.output[0]

    
    def get_action(self,inputs):
        output=self.forward(inputs)
        return np.argmax(output)
    
    def mutate(self,rate=0.1):
        #mutation
        def mut(val):
            if np.random.rand()<rate:
                return val+np.random.randn()*0.5
            return val
        self.w1=np.array([[mut(v) for v in row]for row in self.w1])
        self.b1=np.array([[mut(v) for v in row]for row in self.b1])
        self.w2=np.array([[mut(v) for v in row]for row in self.w2])
        self.b2=np.array([[mut(v) for v in row]for row in self.b2])
        self.w3=np.array([[mut(v) for v in row]for row in self.w3])
        self.b3=np.array([[mut(v) for v in row]for row in self.b3])

    
    def copy(self):
        nn=neuralnetwork(self.input_size,self.hidden_size1,self.hidden_size2,self.output_size)
        nn.w1=self.w1.copy()
        nn.b1=self.b1.copy()
        nn.w2=self.w2.copy()
        nn.b2=self.b2.copy()
        nn.w3=self.w3.copy()
        nn.b3=self.b3.copy()
        return nn

    
    @staticmethod
    def crossover(p1,p2):
        #crossover
        child=neuralnetwork(p1.input_size,p1.hidden_size1,p1.hidden_size2,p1.output_size)
        mask1=np.random.rand(p1.input_size,p1.hidden_size1)<0.5
        mask2=np.random.rand(p1.hidden_size1,p1.hidden_size2)<0.5
        mask3=np.random.rand(p1.hidden_size2,p1.output_size)<0.5
        child.w1=np.where(mask1,p1.w1,p2.w1)
        child.w2=np.where(mask2,p1.w2,p2.w2)
        child.w3=np.where(mask3,p1.w3,p2.w3)
        child.b1=np.where(np.random.rand(1,p1.hidden_size1)<0.5,p1.b1,p2.b1)
        child.b2=np.where(np.random.rand(1,p1.hidden_size2)<0.5,p1.b2,p2.b2)
        child.b3=np.where(np.random.rand(1,p1.output_size)<0.5,p1.b3,p2.b3)
        return child

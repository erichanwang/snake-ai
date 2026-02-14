import numpy as np

class neuralnetwork:
    def __init__(self, input_size=6, hidden_size=16, output_size=4):
        #input: head_x, head_y, tail_x, tail_y, apple_x, apple_y
        #output: up, down, left, right
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        
        #weights - smaller init for better stability
        self.w1=np.random.randn(input_size,hidden_size)*0.1
        self.b1=np.zeros((1,hidden_size))
        self.w2=np.random.randn(hidden_size,output_size)*0.1
        self.b2=np.zeros((1,output_size))

    
    def sigmoid(self,x):
        return 1/(1+np.exp(-np.clip(x,-500,500)))
    
    def relu(self,x):
        return np.maximum(0,x)
    
    def softmax(self,x):
        exp_x=np.exp(x-np.max(x,axis=1,keepdims=True))
        return exp_x/np.sum(exp_x,axis=1,keepdims=True)
    
    def forward(self,x):
        x=np.array(x).reshape(1,-1)
        #input to hidden
        self.z1=np.dot(x,self.w1)+self.b1
        self.a1=self.relu(self.z1)
        #hidden to output
        self.z2=np.dot(self.a1,self.w2)+self.b2
        self.output=self.softmax(self.z2)
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
    
    def copy(self):
        nn=neuralnetwork(self.input_size,self.hidden_size,self.output_size)
        nn.w1=self.w1.copy()
        nn.b1=self.b1.copy()
        nn.w2=self.w2.copy()
        nn.b2=self.b2.copy()
        return nn
    
    @staticmethod
    def crossover(p1,p2):
        #crossover
        child=neuralnetwork(p1.input_size,p1.hidden_size,p1.output_size)
        mask1=np.random.rand(p1.input_size,p1.hidden_size)<0.5
        mask2=np.random.rand(p1.hidden_size,p1.output_size)<0.5
        child.w1=np.where(mask1,p1.w1,p2.w1)
        child.w2=np.where(mask2,p1.w2,p2.w2)
        child.b1=np.where(np.random.rand(1,p1.hidden_size)<0.5,p1.b1,p2.b1)
        child.b2=np.where(np.random.rand(1,p1.output_size)<0.5,p1.b2,p2.b2)
        return child

# MachineLearning  
Number of States: 270.400  
Number of Actions: 6  

needs approx. 200MB additional disk space for qtable and transition matrix.  

# Usage  
## main.py    
starts learning process with specified parameters  

main.py numEpisodes learningRate discountRate decayRate  

if you specify no parameters, the parameters from defines.py will be used  



##### Parameters  
numEpisodes:    Number of Episodes to learn from.  
learningRate:   (0, 1] The higher the faster the agent learns.  
discountRate:   (0, 1] The higher the more longterm focused the agent behaves.   
decayRate:      Controls Exploration-Exploitation tradeoff. The lower the more exploration the agent does.  

## test.py  
lets you watch the agent play in the environment with given qtable. 

test.py qtable  


##### Parameters
qtable:         Path to pickled qtable. Omit .pkl file ending.  


## defines.py    

customize output, learning and environment parameters.  

# Dependencies  
gym 0.12.1  
numpy 1.16.2  
pathos 0.2.3  
pygame 1.9.5 (only for test.py)  

# TODO: 
# Dynamical Model
    # Planners like RRT gives you plan in the global frame.
    # Dynamic model will be in the robot's body frame.
    # Need to map the local frame to the global frame to compute the error.

# Horizon Period
    # x_(k+1) = A_d x_k + B_d U_k
    # y_k = C_d x_k + 0
    # Horizon length (H = 5?)

# Cost Function
    # CEM
    
# Figure out CEM

# Questions:
# What is UR5's dynamical model? Included in the URDF? What are the matrices?

# Task: Make UR5 follow a trajectory. 

# torch.normal(desired mean [n-dimensional vector], stdev, size)





# Define our constant G (number of samples)
# 
# Define our constant H (the horizon length)
# 
# Define our constant K (numbers of action lists to keep) Needs tuning! Too big, you include bad samples, too small, 
# you get stuck at local minimum.
# 
# Define our constant T (times to update mean and variance for the distribution)
# 
# Define initial mu as 0
# 
# Define initial variance as the identity matrix
# 
# Cross-Entropy Method (CEM)
# 
# For env_steps=1 to n:
# 
#   1. Sample G initial plans from some gaussian distribution with some zero mean and some stdev.
#   Each plan should have horizon lenghth H. (G is user defined. Could be up to 1000.)
#   If H=5, you sample 5 random actions using gaussian distribution.
# 
#       For t=1 to T: // This for loop is used for optimizing our gaussian distribution so that samples generated 
#       minimizes our cost.
#         
#           2. Get initial state - call the system (for each joint?)
# 
#           4a. Directly modify your action sequence using Gradient optimization. It takes your generated action 
#           sequences, cost, and "back propagation" and returns a better action sequence. Done through training a graph 
#           neural network to learn the "images" of our robots. (Milestone 3 material)
# 
#           4b. Get H theoretical future states by calling your dynamical model and passing in the list of sampled 
#           actions and the initial state.
# 
#           5. Calculate the cost at each state and sum them for each G.
# 
#           6. Sort your randomly sampled actions based on the sum of cost of each G.
# 
#           7. Pick your top K samples (elite samples). Calculate mean and variance of the action column vectors at 
#           each step from elite samples.
# 
#           8. Replace bottom G-K action sequenses with the newly generated actions using the mean and variance    
#           calculated above.
# 
#       9. Execute the first action from the best action sequence.

# We set trajectory to be (x-cx)^2+(y-cy)^2 = r^2
# Discretize it into steps.
# Can further discretize each step into H steps. 
# Or simply use every H steps as the next H steps if you lower accuracy is acceptable.

# Need to encode the current jointStates
# Need to define cost function that takes current jointStates, action sequences, and future states, 

# Need helper function for calculate mean.
# Need helper function for calculate variance.
# Need helper function for generating k action sequences for H horizon lengths.

# Could add collisionCheck
# Could add steerTo
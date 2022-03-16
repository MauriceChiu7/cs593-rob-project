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
    # Start with an easy quadratic one?
    
# Cross-Entropy Method? 

# Questions:
# What is UR5's dynamical model? Included in the URDF? What are the matrices?
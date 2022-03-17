import pybullet as p
import pybullet_data
import numpy as np
import torch
import csv
import os

import ur5util as ur5
# from ur5pybullet import ur5

END_EFFECTOR_INDEX = 7 # The end effector link index

def sampleNormDistr(mu, sigma):
    mins, maxes = np.array(jointsRange).T

    rnd = torch.normal(mean=mu, std=sigma)
    return torch.clamp(rnd, torch.tensor(mins), torch.tensor(maxes))

def applyAction(handy, jointIds, action):
    # Generate states for each action in each plan
    p.setJointMotorControlArray(handy, jointIds, p.POSITION_CONTROL, action)
    # print("Action: \n", plans[g][h])
    for _ in range(10):
        p.stepSimulation()

def getStateSeq(action_sequence, H, handy, jointIds, currentID):
    state_sequence = []
    for h in range(H):
        applyAction(handy, jointIds, action_sequence[h])
        s = ur5.getCurrJointsState(p, handy)
        state_sequence.append(s)
    # Restore back to original state to run the plan again
    p.restoreState(currentID)
    return state_sequence

def getStateCost(state, prevState):
    # TODO
    # print(f'curr: {state}')
    # print(f'prev: {prevState}')
    # exit(0)
    return 0

def getPathCost(stateSet, s0):
    cost = 0
    for x in range(len(stateSet)):
        if x == 0:
            cost += getStateCost(stateSet[x], s0)
        else:
            cost += getStateCost(stateSet[x], stateSet[x-1])
    return cost

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

def main():
    p.connect(p.GUI)

    ## Loads a plane into the environment
    # plane = p.loadURDF("plane.urdf") # This can be used if you are working in the same directory as your pybullet library
    plane = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0.1]) # Otherwise, this is needed

    ## Setting up the environment
    p.setGravity(0, 0, -9.8)

    urdfFlags = p.URDF_USE_SELF_COLLISION
    ## Loads the UR5 into the environment
    path = f"{os.getcwd()}/ur5pybullet"
    os.chdir(path) # Needed to change directory to load the UR5

    # robot = ur5()
    # robot.






    handy = p.loadURDF(os.path.join(os.getcwd(), "./urdf/real_arm.urdf"), [0.0,0.0,0.0], p.getQuaternionFromEuler([0,0,0]), flags = p.URDF_USE_INERTIA_FROM_FILE)

    # Enable collision for all the link pairs.
    for l0 in range(p.getNumJoints(handy)):
        for l1 in range(p.getNumJoints(handy)):
            if (not l1>l0):
                enableCollision = 1
                # print("collision for pair",l0,l1, p.getJointInfo(handy,l0)[12],p.getJointInfo(handy,l1)[12], "enabled=",enableCollision)
                p.setCollisionFilterPair(handy, handy, l1, l0, enableCollision)

    
    linkStates = ur5.getLinkState(p, handy, END_EFFECTOR_INDEX, 1, verbose=False)
    (jointInfo, JointState) = ur5.getJointsInfo(p, handy, verbose=False)
    
    currState = ur5.getCurrJointsState(p, handy)
    global jointsRange
    jointsRange = ur5.getJointRange(p, handy)

    ####### Milestone 2 ######


    # Define our constant G (number of samples)
    # Define our constant H (the horizon length)
    # Define our constant K (numbers of action lists to keep) Needs tuning! Too big, you include bad samples, too small, 
    #   you get stuck at local minimum.
    # Define our constant T (times to update mean and variance for the distribution)
    G = 30
    H = 5
    N = 300
    T = 5
    k = int(0.4 * G)    # Choosing the top k paths to create new distribution

    currentID = p.saveState()

    # Define initial mu as 0
    # Define initial variance as the identity matrix
    mu = torch.tensor([[0.]*8]*H)
    sigma = torch.tensor([[0.2]*8]*H)
    count = 0

    # Cross-Entropy Method (CEM)
    finalActions = []

    # For env_steps=1 to n:
    for _ in range(N):
        #   1. Sample G initial plans from some gaussian distribution with some zero mean and some stdev.
        #   Each plan should have horizon lenghth H. (G is user defined. Could be up to 1000.)
        #   If H=5, you sample 5 random actions using gaussian distribution.
        plans = [sampleNormDistr(mu, sigma) for _ in range(G)]
        
        #   2. Get initial state - call the system (for each joint?)
        s0 = ur5.getCurrJointsState(p, handy)

        # For t=1 to T: // This for loop is used for optimizing our gaussian distribution so that samples generated 
        #   minimizes our cost.
        for _ in range(T):
            # 4a. Directly modify your action sequence using Gradient optimization. It takes your generated action 
            #   sequences, cost, and "back propagation" and returns a better action sequence. Done through training a graph 
            #   neural network to learn the "images" of our robots. (Milestone 3 material)


            # 4b. Get H theoretical future states by calling your dynamical model and passing in the list of sampled 
            #   actions and the initial state.
            pathSet = []
            jointIds = [0, 1, 2, 3, 4, 5, 6, 7]
            for g in range(G):
                state_sequence = getStateSeq(plans[g], H, handy, jointIds, currentID)
                pathSet.append((state_sequence, plans[g]))

            # 5. Calculate the cost at each state and sum them for each G.
            actionSetCosts = []
            for path in pathSet:
                cost = getPathCost(path[0], s0)
                actionSetCosts.append((path[1],cost))
            
            # 6. Sort your randomly sampled actions based on the sum of cost of each G.
            sortedList = sorted(actionSetCosts, key = lambda x: x[1])

            # 7. Pick your top K samples (elite samples). Calculate mean and variance of the action column vectors at 
            #   each step from elite samples.
            topKPath = [np.array(i[0]) for i in sortedList[0:k]]
            a = np.average(np.array(topKPath), axis=0).tolist()
            b = np.std(np.array(topKPath), axis=0).tolist()
            mu = torch.tensor(a)
            sigma = torch.tensor(b)

            # 8. Replace bottom G-K action sequenses with the newly generated actions using the mean and variance    
            #   calculated above.
            bottomActions = []
            for _ in range(G-k):
                actions = sampleNormDistr(mu, sigma)
                path = getStateSeq(actions, H, handy, jointIds, currentID)
                cost = getPathCost(path, s0)
                bottomActions.append((actions, cost))
            
            topActions = sortedList[0:k]
            actionSet = bottomActions + topActions
            actionSet = sorted(actionSet, key = lambda x: x[1])
        # 9. Execute the first action from the best action sequence.
        bestAction = actionSet[0][0][0]
        applyAction(handy, jointIds, bestAction)
        finalActions.append(bestAction)

        currentID = p.saveState()

        print("Iteration: ", count)
        print("This is the ID: ", currentID)
        count += 1

    # Write to final
    filename = "final_actions.csv"
        
    # writing to csv file
    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        # writing the data rows
        csvwriter.writerows(finalActions)
main()


# 
# 
# 
# 
# 
# 
# 
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
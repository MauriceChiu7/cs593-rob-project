import pybullet as p
import pybullet_data
import numpy as np
import torch
import csv
import os
import math

import ur5util as ur5
# from ur5pybullet import ur5

END_EFFECTOR_INDEX = 7 # The end effector link index
DISCRETIZATION_STEP = 0.2

def diff(v1, v2):
    """
    Computes the difference v1 - v2, assuming v1 and v2 are both vectors
    v2 = [2, 5, 7]
    v1 = [1, 2, 3]
    list(zip(v1, v2)) -> [(1, 2), (2, 5), (3, 7)]
    print([x for x in range(4)]) -> [0, 1, 2, 3]
    """
    return [x1 - x2 for x1, x2 in zip(v1, v2)]

def magnitude(v):
    """
    Computes the magnitude of the vector v.
    """
    return math.sqrt(sum([x*x for x in v]))

def dist(p1, p2):
    """
    Computes the Euclidean distance (L2 norm) between two points p1 and p2
    """
    return magnitude(diff(p1, p2))

# # returns a list of sample from normal distribution with mu and sigma that's between jointMinLimit and jointMaxLimit with length matching mu and sigma.
# def sampleNormDistr(mu, sigma):
#     mins, maxes = np.array(jointsRange).T
#     rnd = torch.normal(mean=mu, std=sigma)
#     return torch.clamp(rnd, torch.tensor(mins), torch.tensor(maxes))

# returns a sample from norma distribution with mu and sigma that's between jointMinLimit and jointMaxLimit
def actionSeqSetFromNormalDist(mu, sigma, numOfPlans, horiLen):
    mins, maxes = np.array([(-3.14159265359, 3.14159265359), (-3.14159265359, 3.14159265359), (-3.14159265359, 3.14159265359), (-3.14159265359, 3.14159265359), (-3.14159265359, 3.14159265359), (-3.14159265359, 3.14159265359), (-0.0, 0.04), (-0.04, 0.0)]).T
    actionSeqSet = [] # action sequence set that contains g sequences
    for g in range(numOfPlans):
        actionSeq = [] # action sequence that contains h actions
        for h in range(horiLen):
            action = []
            for j in range(len(ur5.ACTIVE_JOINTS)):
                sample = torch.normal(mean=mu[h], std=sigma[h])
                action.append(torch.clamp(sample, torch.tensor(mins[j]), torch.tensor(maxes[j])))
            actionSeq.append(action)
        actionSeqSet.append(actionSeq)

    # rnd = torch.normal(mean=mu, std=sigma)
    # return torch.clamp(rnd, torch.tensor(mins), torch.tensor(maxes))
    return actionSeqSet

# # generates a list of H random actions where H is the horizon length
# def genActionSeq(horiLen, mu, sigma):
#     actionSeq = [sampleNormDistr(mu, sigma) for _ in range(horiLen)]
#     return actionSeq

# # generates a list of G plans where G is number of plans
# def genPlans(nPlans, horiLen, mu, sigma):
#     plans = [genActionSeq(horiLen, mu, sigma) for _ in range(nPlans)]
#     return plans

def applyAction(uid, jointIds, action):
    # Generate states for each action in each plan
    p.setJointMotorControlArray(uid, jointIds, p.POSITION_CONTROL, action)
    # print("Action: \n", plans[g][h])
    for _ in range(10):
        p.stepSimulation()

# def getStateSeq(actionSeq, H, uid, jointIds, currentID):
def getStateSeq(actionSeq, uid, jointIds):
    stateSeq = []
    for action in actionSeq:
        p.setJointMotorControlArray(uid, jointIds, p.POSITION_CONTROL, action)
        eePos = p.getLinkState(uid, END_EFFECTOR_INDEX, 1)[0]
        stateSeq.append(eePos)
    return stateSeq

# """
# currState: the robot's current state (position of 8 joints)
# """
# def getCostSeq(currState, actionSeq, plannedStates):
#     # apply action sequence to get state sequences
#     futureStates = []
#     for a in actionSeq:
#         futureStates.append(currState + a)

#     costs = []
#     for i in range(len(futureStates)):
#         costs.append()

#     # calculate distance between each planned states and each estimated states
#     # return a list of costs
#     pass

# def getStateCost(currState, futureState):
#     # TODO
#     return 0

def getStateSequenceCost(stateSeq, hFutureDest):
    cost = 0
    for i in range(len(stateSeq)):
        cost += dist(stateSeq[i], hFutureDest[i])
    return cost
    # cost = 0
    # for x in range(len(stateSet)):
    #     if x == 0:
    #         cost += getStateCost(stateSet[x], s0)
    #     else:
    #         cost += getStateCost(stateSet[x], stateSet[x-1])
    # return cost

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

    handy = p.loadURDF(os.path.join(os.getcwd(), "./urdf/real_arm.urdf"), [0.0,0.0,0.0], p.getQuaternionFromEuler([0,0,0]), flags = p.URDF_USE_INERTIA_FROM_FILE)

    # Enable collision for all the link pairs.
    for l0 in range(p.getNumJoints(handy)):
        for l1 in range(p.getNumJoints(handy)):
            if (not l1>l0):
                enableCollision = 1
                # print("collision for pair",l0,l1, p.getJointInfo(handy,l0)[12],p.getJointInfo(handy,l1)[12], "enabled=",enableCollision)
                p.setCollisionFilterPair(handy, handy, l1, l0, enableCollision)
    
    global jointsRange
    jointsRange = ur5.getJointRange(p, handy)

    trajX = [(0.19 + 0.02 * np.cos(theta * 4)) * np.cos(theta) for theta in np.arange(-np.pi, np.pi, 0.001)]
    trajY = [(0.19 + 0.02 * np.cos(theta * 4)) * np.sin(theta) for theta in np.arange(-np.pi, np.pi, 0.001)]
    trajZ = [2 for z in np.arange(-np.pi, np.pi, 0.001)]
    traj = list(zip(trajX, trajY, trajZ))

    ####### Milestone 2 ######

    N = len(traj)
    G = 30 # Define our constant G (number of samples)
    H = 5 # Define our constant H (the horizon length)
    T = 5 # Define our constant T (times to update mean and standard deviation for the distribution)

    # Define our constant K (numbers of action lists to keep) Needs tuning! Too big, you include bad samples, too small, you get stuck at local minimum.
    K = int(0.4 * G) # Choosing the top K paths to create new distribution

    # currentID = p.saveState()

    # Define H initial mu as 0
    mu = torch.tensor([0.]*H)
    # mu = torch.tensor([[0.]*8]*H)
    
    # Define H initial standard deviation as the identity matrix
    sigma = torch.tensor([0.]*H) # Find out whether this should be torch.eye(8)
    # sigma = torch.tensor([[0.2]*8]*H) # Find out whether this should be torch.eye(8)
    # sigma = torch.tensor([[0.]*8]*H) # Find out whether this should be torch.eye(8)
    
    count = 0

    # Cross-Entropy Method (CEM)
    finalActions = []

    # For env_steps=1 to n:
    for env_step in range(len(traj)):
        # Before simulating, save state. 
        stateId = p.saveState()

        # Get H future destinations from trajectory
        hFutureDest = []
        for h in range(H):
            if env_step + h > len(traj) - 1:
                hFutureDest.append(traj[(env_step + h)%len(traj)])
            else:
                hFutureDest.append(traj[env_step + h])
        # print(hFutureDest)

        # 1. Sample G initial plans from some gaussian distribution with some zero mean and some stdev.
        # Each plan should have horizon lenghth H. (G is user defined. Could be up to 1000.)
        # If H=5, you sample 5 random actions using gaussian distribution.
        actionSeqSet = actionSeqSetFromNormalDist(mu, sigma, G, H)
        
        # 2. Get initial state - call the system (get EE position)
        # currConfig = ur5.getCurrJointsState(p, handy) // might not be necessary.
        # eeCurrPos = p.getLinkState(handy, END_EFFECTOR_INDEX, 1)[0]

        # For t=1 to T: // This for loop is used for optimizing our gaussian distribution so that samples generated 
        #   minimizes our cost.
        for _ in range(T):
            # 4a. Directly modify your action sequence using Gradient optimization. It takes your generated action 
            #   sequences, cost, and "back propagation" and returns a better action sequence. Done through training a graph 
            #   neural network to learn the "images" of our robots. (Milestone 3 material)

            # 4b. Get H theoretical future states by calling your dynamical model and passing in the list of sampled 
            #   actions and the initial state.
            
            # Every plan g in G has an action sequence and the resulting state sequence.
            stateSeqSet = [] # stores the G state sequences
            for g in range(G):
                stateSeq = getStateSeq(actionSeqSet[g], handy, ur5.ACTIVE_JOINTS) # State sequence stores a list of ee positions
                # stateSeqSet.append(stateSeq)
                stateSeqSet.append((stateSeq, actionSeqSet[g]))

            # 5. Calculate the cost at each state and sum them for each G.
            planCosts = []
            for ss in stateSeqSet:
                cost = getStateSequenceCost(ss[0], hFutureDest) # ss[0] is the state sequence calculated from the action sequence
                planCosts.append((ss[1], cost)) # ss[1] is the action sequence
            
            # 6. Sort your randomly sampled actions based on the sum of cost of each g in G.
            sortedActionSeqSet = sorted(planCosts, key = lambda x: x[1])

            # 7. Pick your top K samples (elite samples). Calculate mean and standard deviation of the action column vectors at 
            #   each step from elite samples.
            topKPath = [np.array(i[0]) for i in sortedActionSeqSet[0:K]]
            a = np.average(np.array(topKPath), axis=0).tolist()
            b = np.std(np.array(topKPath), axis=0).tolist()
            mu = torch.tensor(a)
            sigma = torch.tensor(b)

            # 8. Replace bottom G-K action sequenses with the newly generated actions using the mean and standard deviation calculated above.
            # bottomActions = []
            actionSeqSet = actionSeqSet[:len(actionSeqSet)-K]
            # res = test_list[: len(test_list) - K]
            replacement = actionSeqSetFromNormalDist(mu, sigma, G-K, H)
            actionSeqSet.extend(replacement)
            # for _ in range(G-K):
                # actions = sampleNormDistr(mu, sigma)
                # path = getStateSeq(actions, H, handy, ur5.ACTIVE_JOINTS)
                # cost = getStateSequenceCost(path, s0)
                # bottomActions.append((actions, cost))
            
            # topActions = sortedActionSeqSet[0:K]
            # actionSet = bottomActions + topActions
            # actionSet = sorted(actionSet, key = lambda x: x[1])

        # 9. Execute the first action from the best action sequence.
        bestAction = actionSeqSet[0][0][0]
        
        p.restoreState(stateId)
        applyAction(handy, ur5.ACTIVE_JOINTS, bestAction)
        finalActions.append(bestAction)

        # currentID = p.saveState()

        print("Iteration: ", count)
        print("This is the ID: ", stateId)
        count += 1

    # Write to final
    filename = "final_actions.csv"
    
    # writing to csv file
    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        # writing the data rows
        csvwriter.writerows(finalActions)

if __name__ == '__main__':
    main()


# Could add collisionCheck
# Could add steerTo
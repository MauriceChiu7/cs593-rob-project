import argparse
import csv
from itertools import chain
import math
import numpy as np
import os
import pybullet as p
import pybullet_data
import time
import torch

import ur5util as ur5
# from ur5pybullet import ur5

def diff(v1, v2):
    return [x1 - x2 for x1, x2 in zip(v1, v2)]

def magnitude(v):
    return math.sqrt(sum([x*x for x in v]))

def dist(p1, p2):
    return magnitude(diff(p1, p2))

# returns a list of sample from normal distribution with mu and sigma that's between jointMinLimit and jointMaxLimit with length matching mu and sigma.
def actionSeqSetFromNormalDist(mu, sigma, numOfPlans, horiLen):
    mins, maxes = np.array(jointsRange).T
    actionSeqSet = [] # action sequence set that contains g sequences
    for g in range(numOfPlans):
        samp = np.random.multivariate_normal(mu.numpy(), sigma.numpy()).reshape(horiLen, len(mins))
        samp = np.clip(samp, mins, maxes)
        actionSeqSet.append(torch.tensor(samp.reshape(-1)))
    return torch.stack(actionSeqSet)

def applyAction(uid, jointIds, action):
    # Generate states for each action in each plan
    p.setJointMotorControlArray(uid, jointIds, p.POSITION_CONTROL, action)
    for _ in range(10):
        p.stepSimulation()

def getState(action, uid, jointIds):
    applyAction(uid, jointIds, action)
    eePos = p.getLinkState(uid, END_EFFECTOR_INDEX, 1)[0]
    return eePos

def getStateSeq(actionSeq, uid, jointIds):
    stateSeq = []
    for action in actionSeq:
        stateSeq.append(getState(action, uid, jointIds))
    return stateSeq

def getActionSequenceCost(actionSeq, hFutureDest, uid, jointIds):
    actionSeq2 = actionSeq.reshape(len(hFutureDest), -1) # Get H action sequences
    target_traj_cost = 0
    st = getState(actionSeq2[0], uid, jointIds)
    htarg = hFutureDest[0]
    target_traj_cost += dist(st, htarg)

    # Although we probably should've calculated the cost of every state, this error is proved to be too big and causes mayham.
    # for h in range(H):
    #     st = getState(actionSeq2[h], uid, jointIds)
    #     htarg = hFutureDest[h]
    #     target_traj_cost += dist(st, htarg)

    return target_traj_cost # The action sequence cost

def moveToStartingPose(uid, jointIds):
    for _ in range(100):
        applyAction(uid, jointIds, [-2.6,-1.5,1.7,0,0,0,0,0])

# Constants: 
END_EFFECTOR_INDEX = 7 # The end effector link index
N = 100 # number of environmental steps
G = 20  # number of plans
H = 5   # the horizon length
T = 10  # Define our constant T (times to update mean and standard deviation for the distribution)
K = int(0.4 * G) # numbers of action sequences to keep

def main():
    p.connect(p.GUI)
    plane = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0.1])
    p.setGravity(0, 0, -9.8)
    urdfFlags = p.URDF_USE_SELF_COLLISION
    
    ## Loads the UR5 into the environment
    path = f"{os.getcwd()}/ur5pybullet"
    os.chdir(path) # Needed to change directory to load the UR5
    handy = p.loadURDF(os.path.join(os.getcwd(), "./urdf/real_arm.urdf"), [0.0,0.0,0.0], p.getQuaternionFromEuler([0,0,0]), flags = p.URDF_USE_INERTIA_FROM_FILE)

    # Enable collision for all the link pairs
    for l0 in range(p.getNumJoints(handy)):
        for l1 in range(p.getNumJoints(handy)):
            if (not l1>l0):
                enableCollision = 1
                # print("collision for pair",l0,l1, p.getJointInfo(handy,l0)[12],p.getJointInfo(handy,l1)[12], "enabled=",enableCollision)
                p.setCollisionFilterPair(handy, handy, l1, l0, enableCollision)
    
    global jointsRange
    jointsRange = ur5.getJointRange(p, handy)

    moveToStartingPose(handy, ur5.ACTIVE_JOINTS)
    # exit(0)

    # Setting up trajectory for the arm to follow
    resolution = 0.1
    trajX = [-1 * (5 + 0.0 * np.cos(theta * 4)) * np.cos(theta) for theta in np.arange(-np.pi + 0.2, np.pi - 0.2, resolution)]
    trajY = [-1 * (5 + 0.0 * np.cos(theta * 4)) * np.sin(theta) for theta in np.arange(-np.pi + 0.2, np.pi - 0.2, resolution)]
    trajZ = [5 for z in np.arange(-np.pi + 0.2, np.pi - 0.2, resolution)]
    traj = np.array(list(zip(trajX, trajY, trajZ))) / 10
    print(f"trajectory length: {len(traj)}")

    ####### Milestone 2 ######

    print(f"G = {G}, H = {H}, T = {T}, K = {K}")

    # Initialize mean and stdev
    mu = torch.zeros(H, len(ur5.ACTIVE_JOINTS)).flatten()
    sigma = np.pi * torch.eye(len(mu))
    finalActions = []
    count = 0

    useTrajLength = True # true to use length of trajectory, false to use specified N
    numEnvSteps = 0
    if useTrajLength:
        numEnvSteps = len(traj)
    else:
        numEnvSteps = N

    for env_step in range(numEnvSteps):
        # Before simulating, save state. 
        stateId = p.saveState()

        # Get H future destinations from trajectory
        hFutureDest = []
        for h in range(H):
            if env_step + h > len(traj) - 1:
                hFutureDest.append(traj[(env_step + h)%len(traj)])
            else:
                hFutureDest.append(traj[env_step + h])

        # 1. Sample G initial plans using some gaussian distribution with some mean and some stdev.
        # Each plan should have horizon lenghth H. If H=5, you sample 5 random actions using gaussian distribution.
        actionSeqSet = actionSeqSetFromNormalDist(mu, sigma, G, H)

        # 2. Get initial state - call the system (get EE position) # not necessary.
        # currConfig = ur5.getCurrJointsState(p, handy)
        # eeCurrPos = p.getLinkState(handy, END_EFFECTOR_INDEX, 1)[0]

        # optimizing our gaussian distributions so that samples generated have minimum costs.
        for _ in range(T):
            # 4a. Directly modify your action sequence using Gradient optimization. It takes your generated action 
            # sequences, cost, and "back propagation" and returns a better action sequence. Done through training a 
            # graph neural network to learn the "images" of our robots. (Milestone 3 material)

            # 4b. Get H theoretical future states by giving your robot the sampled actions
            # 5. Calculate the cost at each state and sum them for each G.
            planCosts = []
            for actionSeq in actionSeqSet:
                p.restoreState(stateId)
                cost = getActionSequenceCost(actionSeq, hFutureDest, handy, ur5.ACTIVE_JOINTS)
                planCosts.append((actionSeq, cost))

            # 6. Sort your randomly sampled actions based on the sum of cost of each g in G.
            sortedActionSeqSet = sorted(planCosts, key = lambda x: x[1])

            # 7. Pick your top K samples (elite samples). Calculate mean and standard deviation of the action column vectors at each step from elite samples.
            eliteActionSeqSet = []
            for es in range(K):
                eliteActionSeqSet.append(sortedActionSeqSet[es][0])
            eliteActionSeqSet = torch.stack(eliteActionSeqSet)

            mu = torch.mean(eliteActionSeqSet, dim=0)
            sigma = torch.cov(eliteActionSeqSet.T)
            sigma += .02 * torch.eye(len(mu)) # add a small amount of noise to the diagonal to replan to next target

            # 8. Replace bottom G-K action sequenses with the newly generated actions using the mean and standard deviation calculated above.
            replacementSet = actionSeqSetFromNormalDist(mu, sigma, G-K, H)
            actionSeqSet = torch.cat((eliteActionSeqSet, replacementSet))
            
        # 9. Execute the first action from the best action sequence.
        bestAction = actionSeqSet[0][:len(ur5.ACTIVE_JOINTS)]
        # print("=== bestAction ===")
        # print(f'best cost: {getActionSequenceCost(actionSeqSet[0], hFutureDest, handy, ur5.ACTIVE_JOINTS)}')
        # print(bestAction)

        p.restoreState(stateId)
        applyAction(handy, ur5.ACTIVE_JOINTS, bestAction)
        finalActions.append(bestAction.tolist())

        print("Iteration: ", count)
        # print()
        count += 1

    # Write final actions to file
    filename = "../ur5_final_actions.csv"
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(finalActions)

def playback():
    filename = "./ur5_final_actions.csv"
    file = open(filename)
    csvreader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
    finalActions = []
    for row in csvreader:
        finalActions.append(row)
    # print(finalActions)
    file.close()

    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setRealTimeSimulation(0)
    p.setTimeStep(1./500)
    plane = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0.1])
    urdfFlags = p.URDF_USE_SELF_COLLISION
    
    ## Loads the UR5 into the environment
    path = f"{os.getcwd()}/ur5pybullet"
    os.chdir(path) # Needed to change directory to load the UR5
    handy = p.loadURDF(os.path.join(os.getcwd(), "./urdf/real_arm.urdf"), [0.0,0.0,0.0], p.getQuaternionFromEuler([0,0,0]), flags = p.URDF_USE_INERTIA_FROM_FILE)

    # Enable collision for all the link pairs
    for l0 in range(p.getNumJoints(handy)):
        for l1 in range(p.getNumJoints(handy)):
            if (not l1>l0):
                enableCollision = 1
                # print("collision for pair",l0,l1, p.getJointInfo(handy,l0)[12],p.getJointInfo(handy,l1)[12], "enabled=",enableCollision)
                p.setCollisionFilterPair(handy, handy, l1, l0, enableCollision)
    
    moveToStartingPose(handy, ur5.ACTIVE_JOINTS)

    for env_step in range(len(finalActions)):
        applyAction(handy, ur5.ACTIVE_JOINTS, finalActions[env_step])
        # time.sleep(1./125.)
        time.sleep(1./25.)

    # while True:
    #     p.stepSimulation()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CS 593-ROB - Project Milestone 2')
    parser.add_argument('-p', '--play', action='store_true', help='Set true to playback the recorded best actions.')

    args = parser.parse_args()
    
    if args.play:
        playback()
    else: 
        main()



# The Pile (where nightmares come from)

# Could add collisionCheck
# Could add steerTo

# def extract(index, list):
#     return [item[index] for item in list]

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

# returns a sample from norma distribution with mu and sigma that's between jointMinLimit and jointMaxLimit
# def actionSeqSetFromNormalDist(mu, sigma, numOfPlans, horiLen):
#     mins, maxes = np.array(jointsRange).T

#     actionSeqSet = [] # action sequence set that contains g sequences
#     for g in range(numOfPlans):
#         actionSeq = [] # action sequence that contains h actions
#         for h in range(horiLen):
#             action = []
#             for j in range(len(ur5.ACTIVE_JOINTS)):
#                 sample = torch.normal(mean=mu[h], std=sigma[h])
#                 action.append(torch.clamp(sample, torch.tensor(mins[j]), torch.tensor(maxes[j])))
#
#             actionSeq.append(action)
#         actionSeqSet.append(actionSeq)
#     return actionSeqSet

# # generates a list of H random actions where H is the horizon length
# def genActionSeq(horiLen, mu, sigma):
#     actionSeq = [sampleNormDistr(mu, sigma) for _ in range(horiLen)]
#     return actionSeq

# # generates a list of G plans where G is number of plans
# def genPlans(nPlans, horiLen, mu, sigma):
#     plans = [genActionSeq(horiLen, mu, sigma) for _ in range(nPlans)]
#     return plans

# def getStateSequenceCost(stateSeq, hFutureDest):
#     cost = 0
#     for i in range(len(stateSeq)):
#         cost += dist(stateSeq[i], hFutureDest[i])
#     return cost

'''
stateSeqSet = [] # stores the G state sequences
# for g in range(G):
for actionSeq in actionSeqSet:
    stateSeq = getStateSeq(actionSeq, handy, ur5.ACTIVE_JOINTS) # State sequence stores a list of ee positions
    stateSeqSet.append((stateSeq, actionSeq))


planCosts = []
for stateSeq, actSeq in stateSeqSet:
    cost = getStateSequenceCost(stateSeq, hFutureDest) # ss[0] is the state sequence calculated from the action sequence
    planCosts.append((actSeq, cost)) # ss[1] is the action sequence
'''
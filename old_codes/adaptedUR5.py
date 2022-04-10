import argparse
from tkinter import ACTIVE, E
import pybullet as p
import pybullet_data
import torch
import time
import numpy as np
import math
import pickle
import os
import random
import matplotlib.pyplot as plt

ACTIVE_JOINTS = [1,2,3,4,5,6,8,9]
END_EFFECTOR_INDEX = 7 # The end effector link index.

def diff(v1, v2):
    return [x1 - x2 for x1, x2 in zip(v1, v2)]

def magnitude(v):
    return math.sqrt(sum([x*x for x in v]))

def dist(p1, p2):
    return magnitude(diff(p1, p2))

"""
Loads pybullet environment with a horizontal plane and earth like gravity.
"""
def loadEnv():
    # if args.verbose: print(f"\nloading environment...\n")
    # p.connect(p.GUI)
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0.1])
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1./50.)
    # p.setRealTimeSimulation(1)
    # if args.verbose: print(f"\n...environment loaded\n")

"""
Loads the UR5 robot.
"""
def loadUR5():
    # if args.verbose: print(f"\nloading UR5...\n")
    path = f"{os.getcwd()}/ur5pybullet"
    os.chdir(path) # Needed to change directory to load the UR5.
    # print(os.path.join(os.getcwd(), "urdf/real_arm.urdf"))
    # exit()
    if os.path.exists(os.path.join(os.getcwd(), "urdf/real_arm.urdf")):
        print(os.path.join(os.getcwd(), "urdf/real_arm.urdf"))
    uid = p.loadURDF(os.path.join(os.getcwd(), "urdf/real_arm.urdf"), [0.0,0.0,0.0], p.getQuaternionFromEuler([0,0,0]), flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION)
    path = f"{os.getcwd()}/.."
    os.chdir(path) # Back to parent directory.
    # Enable collision for all link pairs.
    for l0 in range(p.getNumJoints(uid)):
        for l1 in range(p.getNumJoints(uid)):
            if (not l1>l0):
                enableCollision = 1
                # print("collision for pair",l0,l1, p.getJointInfo(uid,l0)[12],p.getJointInfo(uid,l1)[12], "enabled=",enableCollision)
                p.setCollisionFilterPair(uid, uid, l1, l0, enableCollision)
    jointsForceRange = getJointsForceRange(uid, ACTIVE_JOINTS)
    # if args.verbose: print(f"\n...UR5 loaded\n")
    return uid, jointsForceRange

"""
Gets the maxForce of each joint active joints.
"""
def getJointsMaxForce(uid, jointIds):
    jointsMaxForces = []
    for j in jointIds:
        jointInfo = p.getJointInfo(uid, j)
        jointsMaxForces.append(jointInfo[10])
    return jointsMaxForces

"""
Gets the min and max force of all active joints.
"""
def getJointsForceRange(uid, jointIds):
    forces = getJointsMaxForce(uid, jointIds)
    jointsForceRange = []
    for f in forces:
        mins = -f
        maxs = f
        jointsForceRange.append((mins, maxs))
    return jointsForceRange

def getLimitPos(jointIds, robot):
    mins = []
    maxes = []
    for id in jointIds:
        info = p.getJointInfo(robot, id)
        mins.append(info[8])
        maxes.append(info[9])
    return mins, maxes

"""
Gets the upper and lower positional limits of each joint.
"""
def getJointsRange(uid, jointIds):
    jointsRange = []
    for a in jointIds:
        jointInfo = p.getJointInfo(uid, a)
        jointsRange.append((jointInfo[8], jointInfo[9]))
    return jointsRange

def getState(robotArm):
    eePos = torch.Tensor(p.getLinkState(robotArm, END_EFFECTOR_INDEX, 1)[0])
    return eePos

def getConfig(uid, jointIds):
    jointPositions = []
    for id in jointIds:
        # print(p.getJointState(uid, id)[0])
        jointPositions.append(p.getJointState(uid, id)[0])
    jointPositions = torch.Tensor(jointPositions)
    return jointPositions

def randomInit(uid):
    # Start ur5 with random positions
    jointsRange = getJointsRange(uid, ACTIVE_JOINTS)
    random_positions = []
    for r in jointsRange:
        rand = np.random.uniform(r[0], r[1])
        random_positions.append(rand)
    p.setJointMotorControlArray(uid, ACTIVE_JOINTS, p.POSITION_CONTROL, random_positions)
    # Give it some time to move there
    for _ in range(100):
        p.stepSimulation()
    initState = getConfig(uid, ACTIVE_JOINTS)
    initCoords = torch.Tensor(p.getLinkState(uid, END_EFFECTOR_INDEX, 1)[0])
    p.addUserDebugLine([0,0,0.1], initCoords, [1,0,0])
    return initState, initCoords

def randomGoal():
    # Generate random goal state for UR5
    x = np.random.uniform(-0.7, 0.7)
    y = np.random.uniform(-0.7, 0.7)
    z = np.random.uniform(0.1, 0.7)
    goalCoords = torch.Tensor([x, y, z])
    p.addUserDebugLine([0,0,0.1], goalCoords, [0,0,1])
    return goalCoords

def getReward(action, jointIds, robotArm, goalState):
    # print(action)
    reward = 0
    p.setJointMotorControlArray(robotArm, jointIds, p.POSITION_CONTROL, action)
    
    # Joint 3 is the elbow joint
    elbowBefore = torch.Tensor(np.array(p.getLinkState(robotArm, 3)[0]))
    p.stepSimulation()
    # state = getState(robotArm)
    # w = torch.Tensor([2000,2000,300,300,300,300,2,3000])
    # reward = (w*state).sum().numpy()
    ee_penalty = dist(getState(robotArm), goalState)
    elbow_penalty = dist(torch.Tensor(p.getLinkState(robotArm, 3)[0]), elbowBefore)

    ee_penalty *= 1
    elbow_penalty *= 1

    # print(f"ee_penalty: {ee_penalty}")
    # print(f"elbow_penalty: {elbow_penalty}")
    reward = reward + ee_penalty
    reward = reward + elbow_penalty
    # if state[-1] > 0.25:
    #     reward += 1000
    # print(f"reward: {reward}")
    return reward

def getEpsReward(eps, jointIds, robotArm, Horizon, goalState):
    numJoints = len(jointIds)
    reward = 0
    for h in range(Horizon):
        start = h*numJoints
        end = start + numJoints
        action = eps[start:end]
        reward += getReward(action, jointIds, robotArm, goalState)

        if h == (Horizon-1):
            futureS = start
            futureE = end
            endDist = getState(robotArm).tolist()
        else:
            futureS = end
            futureE = end + numJoints
        
        actionMag = 8 * math.dist(eps[futureS:futureE], action)
        reward += actionMag

        if h == 2:
            startDist = getState(robotArm).tolist()
            
        # print(h)
        # time.sleep(0.2)
    
    if startDist < endDist:
        # print(f"START: {startDist}")
        # print(f"END: {endDist}")
        reward += 10000
    return reward


torch_seed = np.random.randint(low=0, high=1000)
np_seed = np.random.randint(low=0, high=1000)
py_seed = np.random.randint(low=0, high=1000)
torch.manual_seed(torch_seed)
np.random.seed(np_seed)
random.seed(py_seed)

loadEnv()
uid, jointsForceRange = loadUR5()

Iterations = 30
Epochs = 30
Episodes = 100
Horizon = 50
TopKEps = int(0.3*Episodes)
numJoints = len(ACTIVE_JOINTS)
jointMins,jointMaxes = getLimitPos(ACTIVE_JOINTS, uid)
jointMins = jointMins*Horizon
jointMaxes = jointMaxes*Horizon
jointMins = torch.Tensor(jointMins)
jointMaxes = torch.Tensor(jointMaxes)

saveRun = []    # Store for training
saveAction = []
error = []


goalCoords = randomGoal()
initState, initCoords = randomInit(uid)
print()
print("goalCoords\t", goalCoords)
print("initState\t", initState)
print("initCoords\t", initCoords)



# while 1:
#     p.stepSimulation()

# 1. Set Episode to be constant: graph Error v Epoch (x: Epochs Count y: Distance between bestAction and Subgoal)
# 2. Set Epoch to be constant: graph Error v Episode
# 3. Set Epochs and Episodes to be constant: graph Error v TopK (x: Different amount of top K actions, y: Distance between bestAction and Subgoal)
# 4. Set Epoch, Episodes, and TopK constant: graph Error v Horizon Length (x: Different lengths of Horizon Lengths, y: Distance between bestAction and Subgoal)

for iter in range(Iterations):
    print(f"Running Iteration {iter} ...")
    mu = torch.Tensor([0]*(numJoints * Horizon))
    cov = torch.eye(len(mu)) * ((np.pi/2) ** 2)
    # this is what we should be resetting to

    startState = p.saveState()
    # number of episodes to sample
    currEpsNum = Episodes
    # This is the "memory bank" of episodes we are going to use
    epsMem = []
    error_epoch = []
    error_episode = []
    for e in range(Epochs): 
        print(f"Epoch {e}")     
        # initialize multivariate distribution
        distr = torch.distributions.MultivariateNormal(mu, cov)
        # Now we get the episodes
        for eps in range(currEpsNum):
            # reset environment to start state
            p.restoreState(startState)
            # generate episode
            episode = distr.sample()
            # make sure it's valid by clamping with the mins and maxes
            episode = torch.clamp(episode, jointMins, jointMaxes).tolist()
            # get cost of episode
            cost = getEpsReward(episode, ACTIVE_JOINTS, uid, Horizon, goalCoords)
            # print(cost)
            # exit(0)
            # GRAPH
            #TODO: hard_coded value
            # if e == 29:
            #     error_episode.append(cost)
            # if eps == 9:
            #     error_epoch.append(cost)

            # store the episode, along with the cost, in episode memory
            epsMem.append((episode,cost))
            # print((episode,cost))
            # exit(0)
        p.restoreState(startState)
        # Sort the episode memory
        epsMem = sorted(epsMem, key = lambda x: x[1])
        # print(f"TOP {epsMem[0][1]}")
        # print(f"BOTTOM {epsMem[len(epsMem) -1][1]}")
        # error.append(epsMem[0][1])
        # Now get the top K episodes 
        epsMem = epsMem[0:TopKEps]
        # Now just get a list of episodes from these (episode,cost) pairs
        topK = [x[0] for x in epsMem]
        topK = torch.Tensor(topK)
        sortedTopK = torch.sort(topK, dim=0)
        
        error_per_epoch = epsMem[0][1]
        error_epoch.append((e, error_per_epoch))

        # GRAPH
        # if e ==29 and eps == 9:
        #     #Set Epochs and Episodes to be constant: graph Error v TopK 
        #     # (x: en, y: Distance between bestAction and Subgoal)
        #     pickle.dump(sortedTopK[:5], open(f"topK_{iter}.p", "wb"))
        
        # Now grab the means and covariances of these top K 
        mu = torch.mean(topK, axis = 0)
        std = torch.std(topK, axis = 0)
        var = torch.square(std)
        noise = torch.Tensor([0.2]*Horizon*numJoints)
        var = var + noise
        cov = torch.Tensor(np.diag(var))
        currEpsNum = Episodes - TopKEps
    
    
    # exit(0)
    
    # Set Episodes to be constant: graph Error v Epoch
    if not os.path.exists("./ur5_results/"):
        os.makedirs("./ur5_results/")
    with open("./ur5_results/error_epoch.pkl", "wb") as f:
        pickle.dump(error_epoch, f)
    # print(error_epoch)
    # exit(0)
    

    # Save best action
    bestAction = epsMem[0][0][0:numJoints]
    saveAction.append(bestAction)

    # Need to store this for training
    pairs = []
    pairs.extend(getConfig(uid, ACTIVE_JOINTS))
    pairs.extend(bestAction)

    # print(f"Best Action: {bestAction}")

    # Apply action
    p.setJointMotorControlArray(uid, ACTIVE_JOINTS, p.POSITION_CONTROL, bestAction)
    p.stepSimulation()

    # After applying action, append state2
    pairs.extend(getConfig(uid, ACTIVE_JOINTS))
    saveRun.append(pairs)

# GRAPH
# Set Epoch to be constant: graph Error v Episode
# pickle.dump(error_episode, open("error_episode.p", "wb"))



# with open(f"results/{Epochs}graph.pkl", 'wb') as f:
#     pickle.dump(error, f)
# exit()


trainingFolder = "./trainingData/ur5/"
if not os.path.exists(trainingFolder):
    # create directory if not exist
    os.makedirs(trainingFolder)

with open(trainingFolder + "ur5sample.pkl", 'wb') as f:
    pickle.dump(saveRun, f)




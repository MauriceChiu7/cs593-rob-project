import argparse
from tkinter import ACTIVE
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
    p.connect(p.GUI)
    # p.connect(p.DIRECT)
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

def getState(robotArm):
    END_EFFECTOR_INDEX = 7 # The end effector link index.
    eePos = torch.Tensor(p.getLinkState(robotArm, END_EFFECTOR_INDEX, 1)[0])
    return eePos

def getConfig(uid):
    config = []
    for i in range(len(ACTIVE_JOINTS)):
        config.append(torch.Tensor(p.getLinkState(uid, i, 1)[0]))
    return torch.stack(config)

"""
Moves the robots to their starting position.
"""
def moveToStartingPose(uid, jointIds, random_action):
    steps = random.randint(10, 100)
    print(f"steps: {steps}")
    for _ in range(steps):
        p.setJointMotorControlArray(uid, jointIds, p.TORQUE_CONTROL, forces=random_action)
        p.stepSimulation()

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
    reward = reward - dist(getState(robotArm), goalState)
    reward = reward - dist(torch.Tensor(p.getLinkState(robotArm, 3)[0]), elbowBefore)
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

Iterations = 50
Epochs = 10
Episodes = 30
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

# Generate random goal state for UR5
x = np.random.uniform(-0.7, 0.7)
y = np.random.uniform(-0.7, 0.7)
z = np.random.uniform(0.1, 0.7)
goalState = torch.Tensor([x, y, z])
p.addUserDebugLine([0,0,0.1], goalState, [0,0,1])
print(f"goalState: {goalState}")

# ACTIVE_JOINTS = [1,2,3,4,5,6,8,9]
torqueRanges = [70, 120, 66, 55, 53, 53]
# torqueRanges = [80, 120, 80, 60, 60, 60]
random_action = []

for aRange in torqueRanges:
    sign = 1 if random.random() < 0.5 else -1
    random_action.append(aRange*sign)
random_action.append(0.0)
random_action.append(0.0)
print(f"random_action: {random_action}")
# random_action = [j1, j2, j3, j4, j5, j6, 0.0, 0.0]
moveToStartingPose(uid, ACTIVE_JOINTS, random_action)

# while 1:
#     p.stepSimulation()

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
    error_episode = []
    error_epoch = []
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
            cost = getEpsReward(episode, ACTIVE_JOINTS, uid, Horizon, goalState)

            #TODO: hard_coded value
            if e == 29:
                error_episode.append(cost)
            if eps == 9:
                error_epoch.append(cost)
            # store the episode, along with the cost, in episode memory
            epsMem.append((episode,cost))
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
        if e ==29 and eps == 9:
            #Set Epochs and Episodes to be constant: graph Error v TopK 
            # (x: en, y: Distance between bestAction and Subgoal)
            pickle.dump(sortedTopK[:5], open(f"topK_{iter}.p", "wb"))
        # Now grab the means and covariances of these top K 
        mu = torch.mean(topK, axis = 0)
        std = torch.std(topK, axis = 0)
        var = torch.square(std)
        noise = torch.Tensor([0.2]*Horizon*numJoints)
        var = var + noise
        cov = torch.Tensor(np.diag(var))
        currEpsNum = Episodes - TopKEps
    

    #Set Epoch to be constant: graph Error v Episode
    pickle.dump(error_episode, open("error_episode.p", "wb"))
    #Set Episodes to be constant: graph Error v Epoch
    pickle.dump(error_epoch, open("error_epoch.p", "wb"))


    # with open(f"results/{Epochs}graph.pkl", 'wb') as f:
    #     pickle.dump(error, f)
    # exit()

    # Save best action
    bestAction = epsMem[0][0][0:numJoints]
    saveAction.append(bestAction)

    # Need to store this for training
    pairs = []
    pairs.extend(getConfig(uid))
    pairs.extend(bestAction)

    print(f"Best Action: {bestAction}")

    # Apply action
    p.setJointMotorControlArray(uid, ACTIVE_JOINTS, p.POSITION_CONTROL, bestAction)
    p.stepSimulation()

    # After applying action, append state2
    pairs.extend(getConfig(uid))
    saveRun.append(pairs)



trainingFolder = "./trainingData/ur5/"
if not os.path.exists(trainingFolder):
    # create directory if not exist
    os.makedirs(trainingFolder)

with open(trainingFolder + "ur5sample.pkl", 'wb') as f:
    pickle.dump(saveRun, f)




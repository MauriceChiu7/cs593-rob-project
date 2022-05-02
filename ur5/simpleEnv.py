import pybullet as p
import pybullet_data
import os
import torch
import numpy as np
import math
import pickle
import time
import random
import sys

ACTIVE_JOINTS = [1,2,3,4,5,6,8,9] # All of the movable joints of the UR5
END_EFFECTOR_INDEX = 7 # The end effector link index of the UR5.
ELBOW_INDEX = 3 # The elbow link index of the UR5.
DISCRETIZED_STEP = 0.05 # The step size for discretizing a trajectory.

def diff(v1, v2):
    """
    Description: Calculates the difference between two n-vectors
    
    Input:
    :v1 - {Tensor} a n-vector
    :v2 - {Tensor} a n-vector
    
    Returns:
    :torch.sub(v1, v2) - {Tensor} the difference between the two n-vectors
    """
    return torch.sub(v1, v2)

def magnitude(v):
    """
    Description: Calculates the magnitude of a vector
    
    Input:
    :v - {Tensor} a n-vector
    
    Returns:
    :torch.sqrt(torch.sum(torch.pow(v, 2))) - {Tensor} the magnitude of the n-vector
    """
    return torch.sqrt(torch.sum(torch.pow(v, 2)))

def dist(p1, p2):
    """
    Description: Calculates distance between two vectors
    
    Input:     
    :p1 - {Tensor} a n-vector
    :p2 - {Tensor} a n-vector
    
    Returns:
    :magnitude(diff(p1, p2)) - {Tensor} the distance between the two n-vectors
    """
    return magnitude(diff(p1, p2))

def loadEnv():
    """
    Description: Loads pybullet environment with a horizontal plane and earth like gravity.
    """
    p.connect(p.DIRECT)
    # p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0.1])
    p.setGravity(0, 0, -9.8)
    # p.setTimeStep(1./50.)
    p.setTimeStep(1./20/3)

def loadUR5():
    """
    Description: Loads the UR5 robot into the environment
    
    Returns:
    :uid - {Int} body unique id of the robot
    """
    p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=(0,0,0))
    path = f"{os.getcwd()}/../ur5pybullet"
    print(path)

    os.chdir(path) # Needed to change directory to load the UR5.
    uid = p.loadURDF(os.path.join(os.getcwd(), "./urdf/real_arm.urdf"), [0.0,0.0,0.0], p.getQuaternionFromEuler([0,0,0]), flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION)
    path = f"{os.getcwd()}/../ur5"
    os.chdir(path) # Back to parent directory.
    # Enable collision for all link pairs.
    for l0 in range(p.getNumJoints(uid)):
        for l1 in range(p.getNumJoints(uid)):
            if (not l1>l0):
                enableCollision = 1
                # print("collision for pair",l0,l1, p.getJointInfo(uid,l0)[12],p.getJointInfo(uid,l1)[12], "enabled=",enableCollision)
                p.setCollisionFilterPair(uid, uid, l1, l0, enableCollision)
    return uid

def getConfig(uid, jointIds):
    """
    Description: Gets the position (in radians) of the joint(s)
    
    Input:
    :uid - {Int} body unique id
    :jointIds - {List} a list of joint ids which you want to get the position of
    
    Returns:
    :torch.Tensor(config) - {Tensor} a list of joint positions in radians
    """
    config = []
    for id in jointIds:
        # print(p.getJointState(uid, id)[0])
        config.append(p.getJointState(uid, id)[0])
    return torch.Tensor(config)

def getLimitPos(jointIds, uid):
    """
    Description: Gets the upper and lower positional limits of each joint

    Input:
    :jointIds - {List} a List of joint ids
    :uid - {Int} body unique id of the robot

    Returns:    
    :mins - {List} a list of lower ranges
    :maxes - {List} a list of upper ranges
    """
    mins = []
    maxes = []
    for id in jointIds:
        info = p.getJointInfo(uid, id)
        mins.append(info[8])
        maxes.append(info[9])
    return mins, maxes

def getJointsRange(uid, jointIds):
    """
    Description: Gets the upper and lower positional limits of each joint

    Input:
    :uid - {Int} body unique id of the robot
    :jointIds - {List} a list of joint ids

    Returns:    
    :jointsRange - {List} a list of tuples that contains the lower and upper ranges of joints
    """
    jointsRange = []
    for a in jointIds:
        jointInfo = p.getJointInfo(uid, a)
        jointsRange.append((jointInfo[8], jointInfo[9]))
    return jointsRange

def randomInit(uid):
    """
    Description: Sample random positions for every active joints of the robot and make the robot get into that pose

    Input:
    :uid - {Int} body unique id of the robot

    Returns:
    :initState - {List} a list of Tensors that is the position of every active joint.
    :initCoords - {Tensor} the coordinates of the end-effector
    """
    # Start ur5 with random positions
    jointsRange = getJointsRange(uid, ACTIVE_JOINTS)
    random_positions = []
    for r in jointsRange:
        rand = np.random.uniform(r[0], r[1])
        random_positions.append(rand)
    random_positions = torch.Tensor(random_positions)
    applyAction(uid, random_positions)
    initState = getConfig(uid, ACTIVE_JOINTS)
    initCoords = torch.Tensor(p.getLinkState(uid, END_EFFECTOR_INDEX, 1)[0])
    p.addUserDebugLine([0,0,0.1], initCoords, [1,0,0])
    # time.sleep(10)
    return initState, initCoords

def randomGoal():
    """
    Description: Generates a random goal point for the end-effector

    Returns:
    :goalCoords - {Tensor} the goal coordinates for the end-effector
    """
    # Generate random goal state for UR5
    x = np.random.uniform(-0.7, 0.7)
    y = np.random.uniform(-0.7, 0.7)
    z = np.random.uniform(0.15, 0.7)
    goalCoords = torch.Tensor([x, y, z])
    p.addUserDebugLine([0,0,0.1], goalCoords, [0,0,1])
    return goalCoords

def makeTrajectory(initCoords, goalCoords):
    """
    Description: Make two given points in 3-D space into a discretized trajectory

    Input:
    :initCoords - {Tensor} the trajectory's starting point
    :goalCoords - {Tensor} the trajectory's ending point

    Returns: 
    :torch.stack(traj) - {Tensor} a Tensor sequence that contains all the point/coordinates of the trajectory
    """
    distTotal = dist(goalCoords, initCoords)
    diffBtwin = diff(goalCoords, initCoords)
    incrementTotal = torch.div(distTotal, DISCRETIZED_STEP)
    numSegments = int(math.floor(incrementTotal))+1
    stepVector = diffBtwin / numSegments

    # print("initCoords:\t", initCoords)
    # print("goalCoords:\t", goalCoords)
    # print("distTotal:\t", distTotal)
    # print("diffBtwin:\t", diffBtwin)
    # print("incrementTotal:\t", incrementTotal)
    print()
    print("numSegments:\t", numSegments)
    
    traj = []
    for i in range(numSegments+1):
        # print(torch.mul(stepVector, i))
        traj.append(torch.add(initCoords, torch.mul(stepVector, i)))
    
    for pt in range(len(traj)-1):
        p.addUserDebugLine([0,0,0.1],traj[pt],traj[pt+1])
    
    return torch.stack(traj)

def getState(uid):
    """
    Description: returns the coordinates of the end-effector and the elbow joint

    Input:
    :uid - {Int} body unique id of the robot

    Returns:
    :state - {Tensor} a tensor sequence containing the end-effector and elbow joint coordinates
    """
    eePos = p.getLinkState(uid, END_EFFECTOR_INDEX)[0]
    elbowPos = p.getLinkState(uid, ELBOW_INDEX)[0]
    state = torch.Tensor([eePos, elbowPos])
    return state

def getJointPos(uid):
    """
    Description: returns a list of coordinates of every active joints of the robot

    Input:
    :uid - {Int} body unique id of the robot

    Returns:
    :jointPos - {List} a list of joint coordinates
    """
    jointStates = p.getLinkStates(uid, ACTIVE_JOINTS)
    jointPos = []
    for j in jointStates:
        x, y, z = j[0]
        jointPos.append([x, y, z])
    jointPos[1] = torch.sub(torch.Tensor(jointPos[1]),torch.mul(diff(torch.Tensor(jointPos[1]), torch.Tensor(jointPos[4])), 0.3)).tolist()
    return jointPos

def getReward(action, jointIds, uid, target, distToGoal):
    """
    Description: calculates the cost/reward of an action

    Input:
    :action - {Tensor} a tensor sequence containg the position of all active joints
    :jointIds - {List} a list of joint ids
    :uid - {Int} body unique id of the robot
    :target - {Tensor} the next way point for the end-effector
    :distToGoal - {Float} the distance from the current end-effector coordinates to the goal coordinates

    Returns:
    :reward - {Tensor} the calculated cost/reward of the given action
    """
    state = getState(uid)
    applyAction(uid, action)
    next_state = getState(uid)

    distCost = dist(next_state[0], target)
    elbowCost = dist(next_state[1], state[1])
    groundColliCost = 0
    jointPositions = getJointPos(uid)
    jointZs = []
    for pos in jointPositions:
        jointZs.append(pos[2])
        if pos[2] < 0.15:
            groundColliCost += 1

    weight = torch.Tensor([10, 1, 2])
    rawCost = torch.Tensor([distCost, elbowCost, groundColliCost])
    reward = (weight * rawCost).sum().numpy()
    if distCost > distToGoal: 
        reward += 10

    return reward

def getEpsReward(episode, jointIds, uid, Horizon, goalCoords, distToGoal):
    """
    Description: calculates the cost/reward of an entire episode

    Input:
    :episode - {Tensor} a tensor sequence of joint positions sampled from the multivariate normal distribution
    :jointIds - {List} a list of joint ids
    :uid - {Int} body unique id of the robot
    :Horizon - {Int} the horizon length for the MPC-CEM
    :goalCoord - {Tensor} the coordinates of the goal for the end-effector
    :distToGoal - {Tensor} the remaining distance from the end-effector's position to goal

    Returns:
    :reward - {Tensor} the total cost/reward of the entire episode
    """
    numJoints = len(jointIds)
    reward = 0
    for h in range(Horizon):
        start = h * numJoints
        end = start + numJoints
        action = episode[start:end]
        action = torch.Tensor(action)
        # reward += getReward(action, jointIds, uid, futureStates[h])
        reward += getReward(action, jointIds, uid, goalCoords, distToGoal)
    return reward

def applyAction(uid, action):
    """
    Description: make the robot get into pose with the action given

    Input:
    :uid - {Int} body unique id of the robot
    :action - {Tensor} a tensor sequence that contains the position of joints
    """
    p.setJointMotorControlArray(uid, ACTIVE_JOINTS, p.POSITION_CONTROL, action)
    maxSimSteps = 150
    for s in range(maxSimSteps):
        p.stepSimulation()
        currConfig = getConfig(uid, ACTIVE_JOINTS)
        action = torch.Tensor(action)
        currConfig = torch.Tensor(currConfig)
        error = torch.sub(action, currConfig)
        done = True
        for e in error:
            if abs(e) > 0.02:
                done = False
        if done:
            # print(f"reached position: \n{action}, \nwith target:\n{currConfig}, \nand error: \n{error} \nin step {s}")
            break

def main():
    timeStepId = loadEnv()
    uid = loadUR5()
    jointIds = ACTIVE_JOINTS
    # jointsForceIds = setupJointsForceSlider(uid, jointIds)
    # timeStepId, simStepId = setupTimeStepSlider()
    jointsRange = getJointsRange(uid, ACTIVE_JOINTS)

    pairs = []

    # Saving the previous state
    pairs.extend(getConfig(uid, ACTIVE_JOINTS))
    jointPositions = getJointPos(uid)
    for pos in jointPositions:
        pairs.extend(pos)
    pairs.extend(getState(uid)[1])

    print(np.array(pairs))

    # for i in range(100):
    #     random_positions = []
    #     for r in jointsRange:
    #         rand = np.random.uniform(r[0], r[1])
    #         random_positions.append(rand)
    #     applyAction(uid, random_positions)

        # time.sleep(1)

    

    exit()

    while 1:
        action = []
        debugTS = p.readUserDebugParameter(timeStepId)
        p.setTimeStep(debugTS) # 1/60
        debugSS = p.readUserDebugParameter(simStepId)
        for id in jointsForceIds:
            force = p.readUserDebugParameter(id)
            action.append(force)
        applyAction(uid, jointIds, action, int(debugSS))
    
    # N = 1
    # T = 120
    # G = 220
    # K = int(0.4 * G)
    # H = 5

    # # mu = torch.zeros(H, len(jointsForceRange)).flatten()
    # # sigma = (np.pi * 1e05) * torch.eye(len(mu))
    # mu = torch.zeros(1)
    # sigma = (np.pi * 100) * torch.eye(len(mu))
    # print(f"mu: \n{mu}")
    # print(f"sigma: \n{sigma}")

    # targ = [-250]

    # for n in range(N):
    #     actionSeqSet = genActionSeqSetFromNormalDist(mu, sigma, G, H, (-800, 800))
    #     for t in range(T):
    #         planCosts = []
    #         cost = 0
    #         for actionSeq in actionSeqSet:
    #             # ___LINE 5___
    #             # Calculate the cost of the state sequence.
    #             if args.verbose: print(f"\ncurrMu: {actionSeq}, targMu: {targ}")
    #             direction, magnitude = costFunc(actionSeq, targ)
                
    #             planCosts.append((actionSeq, direction, magnitude))

    #         # ___LINE 6___
    #         # Sort action sequences by cost.
    #         sortedActionSeqSet = sorted(planCosts, key = lambda x: x[2])

    #         # ___LINE 7___
    #         # Update normal distribution to fit top K action sequences.
    #         eliteActionSeqSet = []
    #         for eliteActionSeq in range(K):
    #             eliteActionSeqSet.append(sortedActionSeqSet[eliteActionSeq][0])
    #         eliteActionSeqSet = torch.stack(eliteActionSeqSet)

    #         mu = torch.mean(eliteActionSeqSet, dim=0)
    #         mu.add(torch.mul(direction, magnitude))
    #         sigma = torch.cov(eliteActionSeqSet.T)
    #         # sigma += .02 * torch.eye(len(mu)) # add a small amount of noise to the diagonal to replan to next target
    #         if args.verbose: print(f"mu for envStep {n}:\n{mu}")
    #         if args.verbose: print(f"sigma for envStep {n}:\n{sigma}")

    #         # ___LINE 8___
    #         # Replace bottom G-K sequences with better action sequences.
    #         replacementSet = genActionSeqSetFromNormalDist(mu, sigma, G-K, H, (-800, 800))
    #         actionSeqSet = torch.cat((eliteActionSeqSet, replacementSet))
    # pass

def playback():
    pass

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='CS 593-ROB - Project Milestone 2')
    # parser.add_argument('-p', '--play', action='store_true', help='Set true to playback the recorded best actions.')
    # parser.add_argument('-v', '--verbose', action='store_true', help='Logs debug information.')
    # parser.add_argument('-f', '--fast', action='store_true', help='Trains faster without GUI.')
    # parser.add_argument('-r', '--robot', default='ur5', help='Choose which robot, "ur5" or "a1", to simulate or playback actions.')
    # parser.add_argument('-d', '--debug', action='store_true', help='Displays debug information.')
    # parser.add_argument('-G', '--nplan', help='Number of plans to generate.')
    # parser.add_argument('-T', '--train', help='Number of iterations to train.')
    # parser.add_argument('-H', '--horizon', default=5, help='Set the horizon length.')
    # args = parser.parse_args()
    
    # if args.play:
    #     playback()
    # else:
        # test()
    main()
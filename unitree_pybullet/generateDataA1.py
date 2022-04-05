import pybullet as p
import torch
import time
import numpy as np
import math
import pickle
import os
import sys

robotHeight = 0.420393

def loadDog(pos, yaw):
    # class Dog:
    p.connect(p.DIRECT)
    # p.connect(p.GUI)
    plane = p.loadURDF("plane.urdf")
    p.setGravity(0,0,-9.8)
    p.setTimeStep(1./50)
    #p.setDefaultContactERP(0)
    #urdfFlags = p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
    urdfFlags = p.URDF_USE_SELF_COLLISION
    quadruped = p.loadURDF("a1/urdf/a1.urdf",[pos[0],pos[1],0.48],p.getQuaternionFromEuler([0,0,yaw]), flags = urdfFlags,useFixedBase=False)

    #enable collision between lower legs
    # for j in range (p.getNumJoints(quadruped)):
            # print(p.getJointInfo(quadruped,j))

    lower_legs = [2,5,8,11]
    for l0 in lower_legs:
        for l1 in lower_legs:
            if (l1>l0):
                enableCollision = 1
                # print("collision for pair",l0,l1, p.getJointInfo(quadruped,l0)[12],p.getJointInfo(quadruped,l1)[12], "enabled=",enableCollision)
                p.setCollisionFilterPair(quadruped, quadruped, 2,5,enableCollision)

    jointIds=[]
    paramIds=[]

    maxForceId = p.addUserDebugParameter("maxForce",0,100,20)

    for j in range (p.getNumJoints(quadruped)):
        p.changeDynamics(quadruped,j,linearDamping=0, angularDamping=0)
        info = p.getJointInfo(quadruped,j)
        # print(info)
        jointName = info[1]
        jointType = info[2]
        if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
            jointIds.append(j)

    # print(jointIds)

    p.getCameraImage(480,320)
    p.setRealTimeSimulation(0)

    joints=[]
    return maxForceId, quadruped, jointIds

def getLimitPos(jointIds, quadruped):
    mins = []
    maxes = []
    for id in jointIds:
        info = p.getJointInfo(quadruped, id)
        mins.append(info[8])
        maxes.append(info[9])
    return mins, maxes

def getState(goal, quadruped):
    # ideal height for dog to maintain
    global robotHeight
    hips = []
    # goal point for dog to reach
    goalPoint = [goal[0], goal[1], robotHeight]    
    # [FR, FL, RR, RL]
    hipIds = [2,6,10,14]
    for id in hipIds:
        hips.append(p.getLinkState(quadruped, id)[0])
    pitchR = abs(hips[0][2] - hips[2][2])
    pitchL = abs(hips[1][2] - hips[3][2])
    rollF = abs(hips[0][2] - hips[1][2])
    rollR = abs(hips[2][2] - hips[3][2])
    yawR = abs(hips[0][1] - hips[2][1])
    yawL = abs(hips[1][1] - hips[3][1])
    pos = (p.getLinkState(quadruped, 0)[0])
    distance = math.dist(pos, goalPoint)**2
    heightErr = abs(robotHeight - pos[2])
    state = torch.Tensor([pitchR, pitchL, rollF, rollR, yawR, yawL, distance, heightErr])
    return state


def getFinalState(quadruped):
    state = []
    # [FR, FL, RR, RL]
    hipIds = [2,6,10,14]
    for id in hipIds:
        state.extend(p.getLinkState(quadruped, id)[0])
    
    # Get body
    state.extend(p.getLinkState(quadruped, 0)[0])

    return state


def getReward(goal, action, jointIds, quadruped):
    # print(action)
    p.setJointMotorControlArray(quadruped, jointIds, p.POSITION_CONTROL, action)
    p.stepSimulation()
    state = getState(goal, quadruped)
    w = torch.Tensor([2000,2000,300,300,300,300,2,3000])
    reward = (w*state).sum().numpy()
    if state[-1] > 0.25:
        reward += 1000
    return reward

def getEpsReward(goal, eps, jointIds, quadruped, Horizon):
    numJoints = len(jointIds)
    reward = 0
    for h in range(Horizon):
        start = h*numJoints
        end = start + numJoints
        action = eps[start:end]
        reward += getReward(goal, action, jointIds, quadruped)

        if h == (Horizon-1):
            futureS = start
            futureE = end
            endDist = getState(goal, quadruped).tolist()[6]
        else:
            futureS = end
            futureE = end + numJoints
        
        actionMag = 8 * math.dist(eps[futureS:futureE], action)
        reward += actionMag

        if h == 2:
            startDist = getState(goal, quadruped).tolist()[6]
            
        # print(h)
        # time.sleep(0.2)
    
    if startDist < endDist:
        # print(f"START: {startDist}")
        # print(f"END: {endDist}")
        reward += 10000
    return reward

def main(rollout_index):
    np_seed = np.random.randint(low=0, high=1000)
    np.random.seed(np_seed)
    pos_x = np.random.uniform(low=-1, high=1)
    pos_y = np.random.uniform(low=-1, high=1)
    goal_x = np.random.uniform(low=-10, high=10)
    goal_y = np.random.uniform(low=-10, high=10)
    yaw = np.random.uniform(low=0, high=2*np.pi)

    pos = (pos_x, pos_y)
    goal = (goal_x, goal_y)

    print(f"\npos: {pos}, yaw: {yaw}")
    print(f"goal: {goal}\n")

    maxForceId, quadruped, jointIds = loadDog(pos, yaw)

    Iterations = 300
    Epochs = 10
    Episodes = 100
    Horizon = 50

    print(f"\nIterations: {Iterations}, Epochs: {Epochs}, Episodes: {Episodes}, Horizon: {Horizon}\n")
    trainingFolder = f"./trainingData/iter_{Iterations}_epochs_{Epochs}_episodes_{Episodes}_horizon_{Horizon}/"
    print(f"\ntraining data destination: {trainingFolder}\n")
    # Iterations = 2 # N env steps
    # Epochs = 20 # T
    # Episodes = 10 # G
    # Horizon = 10 # H

    TopKEps = int(0.3*Episodes)
    numJoints = len(jointIds)
    jointMins,jointMaxes = getLimitPos(jointIds, quadruped)
    jointMins = jointMins*Horizon
    jointMaxes = jointMaxes*Horizon
    jointMins = torch.Tensor(jointMins)
    jointMaxes = torch.Tensor(jointMaxes)

    # THIS IS TO MAKE THE ROBOT DROP FIRST

    for _ in range(100):
        p.stepSimulation()

    saveRun = []    # Store for training
    saveAction = []
    error = []
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
                cost = getEpsReward(goal, episode, jointIds, quadruped, Horizon)
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
            # Now grab the means and covariances of these top K 
            mu = torch.mean(topK, axis = 0)
            std = torch.std(topK, axis = 0)
            var = torch.square(std)
            noise = torch.Tensor([0.2]*Horizon*numJoints)
            var = var + noise
            cov = torch.Tensor(np.diag(var))
            currEpsNum = Episodes - TopKEps
        
        # with open(f"results/{Epochs}graph.pkl", 'wb') as f:
        #     pickle.dump(error, f)
        # exit()

        # Save best action
        bestAction = epsMem[0][0][0:numJoints]
        saveAction.append(bestAction)

        # Need to store this for training
        pairs = []
        pairs.extend(getFinalState(quadruped))
        pairs.extend(bestAction)

        # Apply action
        p.setJointMotorControlArray(quadruped, jointIds, p.POSITION_CONTROL, bestAction)
        p.stepSimulation()

        # After applying action, append state2
        pairs.extend(getFinalState(quadruped))
        saveRun.append(pairs)


    
    if not os.path.exists(trainingFolder):
        # create directory if not exist
        os.makedirs(trainingFolder)

    with open(trainingFolder + f"sample_{rollout_index}.pkl", 'wb') as f:
        pickle.dump(saveRun, f)

if __name__ == '__main__':
    print(sys.argv)
    
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    
    print(f"\ngenerating paths {start} to {end}...\n")
    for i in range(start, end):
        main(i)
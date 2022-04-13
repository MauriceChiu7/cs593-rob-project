from ast import Pass
import pybullet as p
import torch
import time
import numpy as np
import math
import pickle
import os

robotHeight = 0.420393
finalSAPairs = []
counter = 0


""" THIS LOADS IN THE A1 DOG AND THE PLANE. ENABLES COLLISION AND RETURNS JOINT INFO
YOU CAN SET THE START POSITION AND ORIENTATION OF THE A1 ROBOT HERE """

def loadDog():
    # class Dog:
    p.connect(p.DIRECT)
    plane = p.loadURDF("plane.urdf")
    p.setGravity(0,0,-9.8)
    p.setTimeStep(1./50)
    urdfFlags = p.URDF_USE_SELF_COLLISION
    quadruped = p.loadURDF("a1/urdf/a1.urdf",[0,0,0.48],[0,0,0,1], flags = urdfFlags,useFixedBase=False)

    lower_legs = [2,5,8,11]
    for l0 in lower_legs:
        for l1 in lower_legs:
            if (l1>l0):
                enableCollision = 1
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


"""THIS FUNCTION GIVES US THE MAX AND MIN ANGLES OF THE JOINTS PASSED IN FOR THE A1 DOG"""

def getLimitPos(jointIds, quadruped):
    mins = []
    maxes = []
    for id in jointIds:
        info = p.getJointInfo(quadruped, id)
        mins.append(info[8])
        maxes.append(info[9])
    return mins, maxes


""" THIS FUNCTION GIVES US THE STATE OF THE ROBOT DOG.
IT RETURNS (PITCH, ROLL, YAW, DISTANCE FROM GOAL, AND HEIGHT ERROR)"""

def getState(quadruped):
    # ideal height for dog to maintain
    global robotHeight
    hips = []
    # goal point for dog to reach
    goalPoint = [10,0, robotHeight]    
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


""" THIS FUNCTION GIVES US THE HIP (X,Y,Z) POSITIONS AND THE CENTER (X,Y,Z) POSITION OF THE A1 ROBOT.
WE USE THIS FUNCTION TO RETURN THE STATE TO TRAIN THE NEURAL NET WITH"""

def getFinalState(quadruped):
    state = []
    # [FR, FL, RR, RL]
    hipIds = [2,6,10,14]
    for id in hipIds:
        state.extend(p.getLinkState(quadruped, id)[0])
    
    # Get body
    state.extend(p.getLinkState(quadruped, 0)[0])

    return state


"""
THIS FUNCTION GIVES US THE REWARD OF A STATE. 
YOU CAN ADJUST THE WEIGHTS OF WHATS REWARDED HERE
"""

def getReward(action, jointIds, quadruped):
    # print(action)
    # This is for the purpose of saving data
    pair = []
    s1 = getFinalState(quadruped)
    # print(s1)
    a = action
    pair.extend(s1)
    pair.extend(a)
    
    # apply action
    p.setJointMotorControlArray(quadruped, jointIds, p.POSITION_CONTROL, action)
    p.stepSimulation()

    # for the purpose of saving data
    s2 = getFinalState(quadruped)
    # print(s2)
    pair.extend(s2)
    # global saPairs
    # saPairs.append(pair)

    # calculating rewards
    state = getState(quadruped)
    w = torch.Tensor([2000,2000,300,300,900,900,2,3000])
    reward = (w*state).sum().numpy()
    if state[-1] > 0.25:
        reward += 1000
    return reward, pair


"""
THIS FUNCTION GIVES US THE REWARD OF A PARTICULAR INPUTTED EPISODE.
IT CALLS THE GETREWARD FUNCTION ON EACH STATE IN THE EPISODE TO DO THIS.
"""

def getEpsReward(eps, jointIds, quadruped, Horizon):
    saPairs = []
    numJoints = len(jointIds)
    reward = 0
    for h in range(Horizon):
        start = h*numJoints
        end = start + numJoints
        action = eps[start:end]
        saReward, saPair = getReward(action, jointIds, quadruped)
        reward += saReward
        saPairs.append(saPair)
        if h == (Horizon-1):
            futureS = start
            futureE = end
            endDist = getState(quadruped).tolist()[6]
        else:
            futureS = end
            futureE = end + numJoints
        
        actionMag = 8 * math.dist(eps[futureS:futureE], action)
        reward += actionMag

        if h == 2:
            startDist = getState(quadruped).tolist()[6]
    
    if startDist < endDist:
        reward += 10000
    return reward, saPairs

# ARRAYS TO SAVE DATA THAT WE WANT TO PICKLE

# PARAMETERS TO RUN MPC ON
Iterations = 150
Epochs = 5
Episodes = 50
Horizon = 90
TopKEps = int(0.3*Episodes)
MultIters = 6

# ACTIONS AND SADATA PATHS BASED ON PARAMETERS
actionPath = f"multActions_I{Iterations}_E{Epochs}_Eps{Episodes}/"
saData = f"saData_Mult/"

# CREATE FOLDERS TO SAVE DATA IN (IF NOT ALREADY EXISTING)
if not os.path.exists(actionPath):
    # create directory if not exist
    os.makedirs(actionPath)

if not os.path.exists(saData):
    # create directory if not exist
    os.makedirs(saData)


loadDog()
for multRun in range(MultIters):
    saveAction = []
    p.disconnect()
    maxForceId, quadruped, jointIds = loadDog()
    numJoints = len(jointIds)
    jointMins,jointMaxes = getLimitPos(jointIds, quadruped)
    jointMins = jointMins*Horizon
    jointMaxes = jointMaxes*Horizon
    jointMins = torch.Tensor(jointMins)
    jointMaxes = torch.Tensor(jointMaxes)

    # THIS IS TO MAKE THE ROBOT DROP FIRST
    for _ in range(100):
        p.stepSimulation()

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
        saMem = []
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
                cost, saPairs = getEpsReward(episode, jointIds, quadruped, Horizon)
                # store the episode, along with the cost, in episode memory
                epsMem.append((episode,cost))
                saMem.append((saPairs, cost))
            p.restoreState(startState)
            # Sort the episode memory
            epsMem = sorted(epsMem, key = lambda x: x[1])
            saMem = sorted(saMem, key = lambda x: x[1])
            # Now get the top K episodes 
            epsMem = epsMem[0:TopKEps]
            # We save the top 2 episode's state action pairs as training data for the nueral net
            saMem = saMem[0:2]
            # Now just get a list of episodes from these (episode,cost) pairs
            topK = [x[0] for x in epsMem]
            # This saves the top 2 episodes so that we can play back later
            saveTopKEpsMem1 = topK[0]
            saveTopKEpsMem2 = topK[1]
            topK = torch.Tensor(topK)
            # Now grab the means and covariances of these top K 
            mu = torch.mean(topK, axis = 0)
            std = torch.std(topK, axis = 0)
            var = torch.square(std)
            noise = torch.Tensor([0.2]*Horizon*numJoints)
            var = var + noise
            cov = torch.Tensor(np.diag(var))
            currEpsNum = Episodes - TopKEps
        
        print(len(saveTopKEpsMem1))
        with open(f"{actionPath}MultRun_{multRun}_Iter_{iter}_Top1.pkl", 'wb') as f:
            pickle.dump(saveTopKEpsMem1, f)
        with open(f"{actionPath}MultRun_{multRun}_Iter_{iter}_Top2.pkl", 'wb') as f:
            pickle.dump(saveTopKEpsMem2, f)
        bestAction = epsMem[0][0][0:numJoints]
        saveAction.append(bestAction)

        saTopK = [x[0] for x in saMem]
        for x in saTopK:
            finalSAPairs.extend(x)

        p.setJointMotorControlArray(quadruped, jointIds, p.POSITION_CONTROL, bestAction)
        p.stepSimulation()

        temp = p.saveState()
        p.restoreState(temp)
        # After applying action, append state2
    
    with open(f"{actionPath}run_I{Iterations}_E{Epochs}_Eps{Episodes}_Mult{multRun}.pkl", 'wb') as f:
        pickle.dump(saveAction, f)


print("DONE!!!!!")

with open(f"{saData}MULT{MultIters}_run_I{Iterations}_E{Epochs}_Eps{Episodes}.pkl", 'wb') as f:
    pickle.dump(finalSAPairs, f)
    print(f"WE HAVE {len(finalSAPairs)} DATA POINTS FROM THIS RUN!")
            

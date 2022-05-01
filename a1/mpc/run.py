import pybullet as p
import torch
import time
import numpy as np
import math
import pickle
import os

robotHeight = 0.420393

def loadDog():
    '''
    Description:
    Loads the dog model from the file

    Inputs:
    None

    Returns:
    :quadruped {int}: quadruped model id
    '''
    # class Dog:
    p.connect(p.DIRECT)
    plane = p.loadURDF("../../unitree_pybullet/data/plane.urdf")
    p.setGravity(0,0,-9.8)
    p.setTimeStep(1./50)
    #p.setDefaultContactERP(0)
    #urdfFlags = p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
    urdfFlags = p.URDF_USE_SELF_COLLISION
    quadruped = p.loadURDF("../../unitree_pybullet/data/a1/urdf/a1.urdf",[0,0,0.48],[0,0,0,1], flags = urdfFlags,useFixedBase=False)

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
    '''
    Description:
    Gets the limits of the joints

    Inputs:
    :jointIds {list}: list of joint ids
    :quadruped {int}: quadruped model id

    Returns:
    :limits {list}: list of limits
    '''
    mins = []
    maxes = []
    for id in jointIds:
        info = p.getJointInfo(quadruped, id)
        mins.append(info[8])
        maxes.append(info[9])
    return mins, maxes

def getState(quadruped):
    '''
    Description:
    Gets the state of the quadruped

    Inputs:
    :quadruped {int}: quadruped model id

    Returns:
    :state {list}: list of state
    '''
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


def getFinalState(quadruped):
    '''
    Description:
    Gets the final state of the quadruped

    Inputs:
    :quadruped {int}: quadruped model id

    Returns:
    :state {list}: list of state
    '''
    state = []
    # [FR, FL, RR, RL]
    hipIds = [2,6,10,14]
    for id in hipIds:
        state.extend(p.getLinkState(quadruped, id)[0])
    
    # Get body
    state.extend(p.getLinkState(quadruped, 0)[0])

    return state


def getReward(action, jointIds, quadruped):
    '''
    Description:
    Gets the reward for the action

    Inputs:
    :action {list}: list of action
    :jointIds {list}: list of joint ids
    :quadruped {int}: quadruped model id

    Returns:
    :reward {float}: reward
    '''
    # print(action)
    p.setJointMotorControlArray(quadruped, jointIds, p.POSITION_CONTROL, action)
    p.stepSimulation()
    state = getState(quadruped)
    w = torch.Tensor([2000,2000,300,300,300,300,2,3000])
    reward = (w*state).sum().numpy()
    if state[-1] > 0.25:
        reward += 1000
    return reward

def getEpsReward(eps, jointIds, quadruped, Horizon):
    '''
    Description:
    Gets the reward for the episode

    Inputs:
    :eps {list}: list of action in episode
    :jointIds {list}: list of joint ids
    :quadruped {int}: quadruped model id
    :Horizon {int}: horizon

    Returns:
    :reward {float}: reward
    '''
    numJoints = len(jointIds)
    reward = 0
    for h in range(Horizon):
        start = h*numJoints
        end = start + numJoints
        action = eps[start:end]
        reward += getReward(action, jointIds, quadruped)

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
        # print(h)
        # time.sleep(0.2)
    if startDist < endDist:
        # print(f"START: {startDist}")
        # print(f"END: {endDist}")
        reward += 10000
    return reward

maxForceId, quadruped, jointIds = loadDog()

# TODO: Change paramters for tuning
Iterations = 10
Epochs = 10
Episodes = 10
Horizon = 50

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

# Save for every epoch
graphFolder = "./graphs/"
if not os.path.exists(graphFolder):
    # create directory if not exist
    os.makedirs(graphFolder)

saveAction = []
error = []
for it in range(Iterations):
    print(f"Running Iteration {it} ...")
    mu = torch.Tensor([0]*(numJoints * Horizon))
    cov = torch.eye(len(mu)) * ((np.pi/2) ** 2)
    # this is what we should be resetting to
    startState = p.saveState()
    p.restoreState(startState)
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
            cost = getEpsReward(episode, jointIds, quadruped, Horizon)
            # store the episode, along with the cost, in episode memory
            epsMem.append((episode,cost))
        p.restoreState(startState)
        # Sort the episode memory
        epsMem = sorted(epsMem, key = lambda x: x[1])

        # TODO: Uncomment to track errors
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
    
    # TODO: Uncomment to track errors
    # with open(graphFolder + f"run_I{Iterations}_E{Epochs}_Eps{Episodes}_H{Horizon}_epoch_{it}.pkl", 'wb') as f:
    #     pickle.dump(error, f)

    # Save best action
    bestAction = epsMem[0][0][0:numJoints]
    saveAction.append(bestAction)

    # Apply action
    p.setJointMotorControlArray(quadruped, jointIds, p.POSITION_CONTROL, bestAction)
    p.stepSimulation()
    temp = p.saveState()
    p.restoreState(temp)


folder = "./results/"
if not os.path.exists(folder):
    # create directory if not exist
    os.makedirs(folder)


print("DONE!!!!!")
# Save the actions to replay later
with open(folder + f"run_I{Iterations}_E{Epochs}_Eps{Episodes}_H{Horizon}.pkl", 'wb') as f:
    pickle.dump(saveAction, f)
 

import pybullet as p
import pybullet_data
import os
import torch
import numpy as np
import math
import pickle
import time
import random

ACTIVE_JOINTS = [1,2,3,4,5,6,8,9]
END_EFFECTOR_INDEX = 7 # The end effector link index.
ELBOW_INDEX = 3 # The end effector link index.
DISCRETIZED_STEP = 0.05
CTL_FREQ = 20
SIM_STEPS = 3
GAMMA = 0.9

"""
Calculates the difference between two vectors.
"""
def diff(v1, v2):
    return torch.sub(v1, v2)

"""
Calculates the magnitude of a vector.
"""
def magnitude(v):
    return torch.sqrt(torch.sum(torch.pow(v, 2)))

"""
Calculates distance between two vectors.
"""
def dist(p1, p2):
    return magnitude(diff(p1, p2))

"""
Loads pybullet environment with a horizontal plane and earth like gravity.
"""
def loadEnv():
    p.connect(p.DIRECT) 
    # p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0.1])
    p.setGravity(0, 0, -9.8)
    # p.setTimeStep(1./50.)
    p.setTimeStep(1./CTL_FREQ/SIM_STEPS)

"""
Loads the UR5 robot.
"""
def loadUR5():
    p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=(0,0,0))
    path = f"{os.getcwd()}/../ur5pybullet"
    print(path)
    # exit()
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
    jointPositions = []
    for id in jointIds:
        # print(p.getJointState(uid, id)[0])
        jointPositions.append(p.getJointState(uid, id)[0])
    jointPositions = torch.Tensor(jointPositions)
    return jointPositions

def getLimitPos(jointIds, quadruped):
    mins = []
    maxes = []
    for id in jointIds:
        info = p.getJointInfo(quadruped, id)
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

def randomInit(uid):
    # Start ur5 with random positions
    jointsRange = getJointsRange(uid, ACTIVE_JOINTS)
    random_positions = []
    for r in jointsRange:
        rand = np.random.uniform(r[0], r[1])
        random_positions.append(rand)
    p.setJointMotorControlArray(uid, ACTIVE_JOINTS, p.POSITION_CONTROL, random_positions)
    # Give it some time to move there
    for _ in range(SIM_STEPS):
        p.stepSimulation()
    initState = getConfig(uid, ACTIVE_JOINTS)
    initCoords = torch.Tensor(p.getLinkState(uid, END_EFFECTOR_INDEX, 1)[0])
    p.addUserDebugLine([0,0,0.1], initCoords, [1,0,0])
    # time.sleep(10)
    return initState, initCoords

def randomGoal():
    # Generate random goal state for UR5
    x = np.random.uniform(-0.7, 0.7)
    y = np.random.uniform(-0.7, 0.7)
    z = np.random.uniform(0.1, 0.7)
    goalCoords = torch.Tensor([x, y, z])
    p.addUserDebugLine([0,0,0.1], goalCoords, [0,0,1])
    return goalCoords

def makeTrajectory(initCoords, goalCoords):
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
    eePos = p.getLinkState(uid, END_EFFECTOR_INDEX)[0]
    elbowPos = p.getLinkState(uid, ELBOW_INDEX)[0]
    state = torch.Tensor([eePos, elbowPos])
    return state

def getReward(action, jointIds, uid, target):
    state = getState(uid)

    applyAction(uid, action)

    next_state = getState(uid)

    eeCost = dist(next_state[0], target)
    elbowCost = dist(next_state[1], state[1])

    weight = torch.Tensor([10, 1])
    rawCost = torch.Tensor([eeCost, elbowCost])
    # print("rawCost:\t", rawCost)
    reward = (weight * rawCost).sum().numpy()
    # print("reward:\t\t", reward)
    return reward

def getEpsReward(episode, jointIds, uid, Horizon, futureStates):
    numJoints = len(jointIds)
    reward = 0
    for h in range(Horizon):
        start = h * numJoints
        end = start + numJoints
        action = episode[start:end]
        reward += getReward(action, jointIds, uid, futureStates[h])
    return reward

def applyAction(uid, action):
    p.setJointMotorControlArray(uid, ACTIVE_JOINTS, p.POSITION_CONTROL, action)
    for _ in range(SIM_STEPS):
        p.stepSimulation()

def main():
    torch_seed = np.random.randint(low=0, high=1000)
    np_seed = np.random.randint(low=0, high=1000)
    py_seed = np.random.randint(low=0, high=1000)
    torch.manual_seed(torch_seed)
    np.random.seed(np_seed)
    random.seed(py_seed)

    trainingFolder = "./trainingData/"
    errorFolder = "./error/"
    if not os.path.exists(trainingFolder):
        os.makedirs(trainingFolder)
    if not os.path.exists(errorFolder):
        os.makedirs(errorFolder)
    
    loadEnv()
    uid = loadUR5()
    jointsRange = getJointsRange(uid, ACTIVE_JOINTS)

    goalCoords = randomGoal()
    initState, initCoords = randomInit(uid)

    debug = {
        'goalCoords': goalCoords,
        'initState': initState,
        'initCoords': initCoords
    }

    traj = makeTrajectory(initCoords, goalCoords)
    print("initCoords:\t", initCoords)
    print("initState:\t", initState)
    print("goalCoords:\t", goalCoords)
    print("traj:\n", traj)
    
    # Constants:
    Iterations = len(traj) # N - envSteps
    Epochs = 40 # T - trainSteps
    Episodes = 200 # G - plans
    Horizon = 10 # H - horizonLength
    TopKEps = int(0.3*Episodes)
    jointMins,jointMaxes = getLimitPos(ACTIVE_JOINTS, uid)
    jointMins = jointMins*Horizon
    jointMaxes = jointMaxes*Horizon
    jointMins = torch.Tensor(jointMins)
    jointMaxes = torch.Tensor(jointMaxes)

    saveRun = []
    saveAction = []

    finalEePos = []
    for envStep in range(Iterations):
        print(f"Running Iteration {envStep} ...")

        mu = torch.Tensor([0]*(len(ACTIVE_JOINTS) * Horizon))
        cov = torch.eye(len(mu)) * ((np.pi/2) ** 2)
        startState = p.saveState()
        p.restoreState(startState)
        
        stateId = p.saveState()

        futureStates = []
        for h in range(Horizon):
            if envStep + h > len(traj) - 1:
                futureStates.append(traj[-1])
            else:
                futureStates.append(traj[envStep + h])
        futureStates = torch.stack(futureStates)
        # print(envStep)
        # print("futureStates:\n", futureStates)
        epsMem = []
        for e in range(Epochs):
            print(f"Epoch {e}")
            distr = torch.distributions.MultivariateNormal(mu, cov)
            for eps in range (Episodes):
                p.restoreState(stateId)
                episode = distr.sample()
                episode = torch.clamp(episode, jointMins, jointMaxes).tolist()
                cost = getEpsReward(episode, ACTIVE_JOINTS, uid, Horizon, futureStates)
                epsMem.append((episode,cost))
            p.restoreState(stateId)
            epsMem = sorted(epsMem, key = lambda x: x[1])
            epsMem = epsMem[0:TopKEps]
            topK = [x[0] for x in epsMem]
            topK = torch.Tensor(topK)
            mu = torch.mean(topK, axis = 0)
            std = torch.std(topK, axis = 0)
            var = torch.square(std)
            noise = torch.Tensor([0.2]*Horizon*len(ACTIVE_JOINTS))
            var = var + noise
            cov = torch.Tensor(np.diag(var))
            currEpsNum = Episodes - TopKEps

        bestAction = epsMem[0][0][0:len(ACTIVE_JOINTS)]
        saveAction.append(bestAction)
        
        pairs = []
        pairs.extend(getConfig(uid, ACTIVE_JOINTS))
        pairs.extend(bestAction)
        
        applyAction(uid, bestAction)

        temp = p.saveState()
        p.restoreState(temp)

        pairs.extend(getConfig(uid, ACTIVE_JOINTS))
        saveRun.append(pairs)

        finalEePos.append(getState(uid)[0].tolist())

    finalEePos = np.array(finalEePos)
    traj = np.array(traj)

    with open(trainingFolder + f"ur5sample.pkl", 'wb') as f:
        pickle.dump(saveRun, f)

    with open(errorFolder + f"debug.pkl", 'wb') as f:
        pickle.dump(debug, f)

    with open(errorFolder + f"finalEePos.pkl", 'wb') as f:
        pickle.dump(finalEePos, f)

    with open(errorFolder + f"traj.pkl", 'wb') as f:
        pickle.dump(traj, f)


    # while 1:
    #     p.stepSimulation()


if __name__ == '__main__':
    main()
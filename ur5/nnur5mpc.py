import os
import torch
from torch import nn
import numpy as np
import math
import pickle
import random
from datetime import datetime

ACTIVE_JOINTS = [1,2,3,4,5,6,8,9]
END_EFFECTOR_INDEX = 7 # The end effector link index.
ELBOW_INDEX = 3 # The end effector link index.
DISCRETIZED_STEP = 0.05
CTL_FREQ = 20
SIM_STEPS = 3
GAMMA = 0.9

def loadNN(args):
    stateLength = 8
    actionLength = 8

    # Load the neural network
    neuralNet = nn.Sequential(
                        nn.Linear(stateLength + actionLength, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Linear(256, stateLength),
                     )
    # Edit this when done with training
    neuralNet.load_state_dict(torch.load(args.model_folder))
    neuralNet.eval()

    return neuralNet, stateLength, actionLength


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
    for _ in range(100):
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


def getStateFromNN(neuralNet, action, initialState):
    # Predict the next state with state1 + action
    state_action = []
    state_action.extend(initialState)
    state_action.extend(action)
    nnPredState = neuralNet.forward(torch.Tensor(state_action))
    return nnPredState

def getReward(action, jointIds, target, neuralNet, initState):
    nnPredState = getStateFromNN(neuralNet, action, initState)

    eeCost = dist(nnPredState[0], target)
    # elbowCost = dist(nnPredState[3], torch.Tensor(initState)[3])
    elbowCost = 0

    weight = torch.Tensor([10, 1])
    rawCost = torch.Tensor([eeCost, elbowCost])
    # print("rawCost:\t", rawCost)
    reward = (weight * rawCost).sum().numpy()
    # print("reward:\t\t", reward)
    return reward

def getEpsReward(episode, jointIds, Horizon, futureStates, neuralNet, initState):
    numJoints = len(jointIds)
    reward = 0
    for h in range(Horizon):
        start = h * numJoints
        end = start + numJoints
        action = episode[start:end]
        reward += getReward(action, jointIds, futureStates[h], neuralNet, initState)
    return reward


def main():
    # Load Neural Network instead of pybullet stuff
    neuralNet, stateLength, actionLength = loadNN()

    torch_seed = np.random.randint(low=0, high=1000)
    np_seed = np.random.randint(low=0, high=1000)
    py_seed = np.random.randint(low=0, high=1000)
    torch.manual_seed(torch_seed)
    np.random.seed(np_seed)
    random.seed(py_seed)

    testRunResults = "./testRunResults/"
    # errorFolder = "./error/"
    if not os.path.exists(testRunResults):
        os.makedirs(testRunResults)
    # if not os.path.exists(errorFolder):
    #     os.makedirs(errorFolder)

    # goalCoords = randomGoal()
    goalCoords = [-0.6484, -0.3258,  0.3040]
    initState = [0] * 8
    initCoords = [-0.8144, -0.1902,  0.0707]

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
    # jointMins,jointMaxes = getLimitPos(ACTIVE_JOINTS, uid)
    jointMins = [-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, 0, -0.04]
    jointMaxes = [np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, 0.04, 0]
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

        futureStates = []
        for h in range(Horizon):
            if envStep + h > len(traj) - 1:
                futureStates.append(traj[-1])
            else:
                futureStates.append(traj[envStep + h])
        futureStates = torch.stack(futureStates)

        epsMem = []
        for e in range(Epochs):
            print(f"Epoch {e}")
            distr = torch.distributions.MultivariateNormal(mu, cov)
            for eps in range (Episodes):
                episode = distr.sample()
                episode = torch.clamp(episode, jointMins, jointMaxes).tolist()
                cost = getEpsReward(episode, ACTIVE_JOINTS, Horizon, futureStates, neuralNet, initState)
                epsMem.append((episode,cost))

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
        
        # applyAction(uid, bestAction)

        # finalEePos.append(getState(uid)[0].tolist())

    # finalEePos = np.array(finalEePos)
    # traj = np.array(traj)

    with open(testRunResults + f"test_{datetime.now()}.pkl", 'wb') as f:
        pickle.dump(saveAction, f)

    with open(testRunResults + f"test.pkl", 'wb') as f:
        pickle.dump(saveAction, f)

    # with open(errorFolder + f"debug.pkl", 'wb') as f:
    #     pickle.dump(debug, f)

    # with open(errorFolder + f"finalEePos.pkl", 'wb') as f:
    #     pickle.dump(finalEePos, f)

    # with open(errorFolder + f"traj.pkl", 'wb') as f:
    #     pickle.dump(traj, f)

    # while 1:
    #     p.stepSimulation()


if __name__ == '__main__':
    main()
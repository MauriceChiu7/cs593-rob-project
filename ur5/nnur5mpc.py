import os
import torch
from torch import nn
import numpy as np
import math
import pickle
import random
from datetime import datetime
import time

ACTIVE_JOINTS = [1,2,3,4,5,6,8,9]
END_EFFECTOR_INDEX = 7 # The end effector link index.
ELBOW_INDEX = 3 # The end effector link index.
DISCRETIZED_STEP = 0.05
CTL_FREQ = 20
SIM_STEPS = 3
GAMMA = 0.9

def loadNN():
    stateLength = 35
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
    modelName = f"UR5_V1_Model_2Layers"
    # create model folder
    modelFolder = f"./mult_models/{modelName}/"
    if not os.path.exists(modelFolder):
        # create directory if not exist
        os.makedirs(modelFolder)
    neuralNet.load_state_dict(torch.load(f"{modelFolder}UR5_V1_Model_2Layers_model1.pt"))
    neuralNet.eval()

    return neuralNet, stateLength, actionLength


"""
Calculates the difference between two vectors.
"""
def diff(v1, v2):
    v1 = torch.Tensor(v1)
    v2 = torch.Tensor(v2)
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
    
    # for pt in range(len(traj)-1):
    #     p.addUserDebugLine([0,0,0.1],traj[pt],traj[pt+1])
    
    return torch.stack(traj)


def getStateFromNN(neuralNet, action, initialState):
    # Predict the next state with state1 + action
    state_action = []
    state_action.extend(initialState)
    state_action.extend(action)
    nnPredState = neuralNet.forward(torch.Tensor(state_action))
    return nnPredState

# def getReward(action, jointIds, target, neuralNet, initState):
#     nnPredState = getStateFromNN(neuralNet, action, initState)

#     eeCost = dist(nnPredState[0], target)
#     # elbowCost = dist(nnPredState[3], torch.Tensor(initState)[3])
#     elbowCost = 0

#     weight = torch.Tensor([10, 1])
#     rawCost = torch.Tensor([eeCost, elbowCost])
#     # print("rawCost:\t", rawCost)
#     reward = (weight * rawCost).sum().numpy()
#     # print("reward:\t\t", reward)
#     return reward

def getReward(action, jointIds, target, distToGoal, state, next_state):
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
    
    # state = getState(uid)
    # applyAction(uid, action)

    # next_state = getState(uid)

    distCost = dist(next_state[23:26], target)
    elbowCost = dist(next_state[14:17], state[14:17])
    groundColliCost = 0

    positions = next_state[51:75]

    jointPositions = []
    for i in range(0, len(positions), 3):
        joint = []
        for j in range(3):
            joint.append(positions[i+j])
        jointPositions.append(joint)

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

# def getEpsReward(episode, jointIds, Horizon, futureStates, neuralNet, initState):
#     numJoints = len(jointIds)
#     reward = 0
#     for h in range(Horizon):
#         start = h * numJoints
#         end = start + numJoints
#         action = episode[start:end]
#         reward += getReward(action, jointIds, futureStates[h], neuralNet, initState)
#     return reward

# def getEpsReward(episode, jointIds, Horizon, futureStates, neuralNet, initState):
def getEpsReward(episode, jointIds, Horizon, goalCoords, distToGoal, neuralNet, initState):
    """
    Description: calculates the cost/reward of an entire episode

    Input:
    :episode - {Tensor} a tensor sequence of joint positions sampled from the multivariate normal distribution
    :jointIds - {List} a list of joint ids
    :Horizon - {Int} the horizon length for the MPC-CEM
    :goalCoord - {Tensor} the coordinates of the goal for the end-effector
    :distToGoal - {Tensor} the remaining distance from the end-effector's position to goal
    :neuralNet - {nn} our policy
    :initState - {List} the initial state of the UR5

    Returns:
    :reward - {Tensor} the total cost/reward of the entire episode
    """
    numJoints = len(jointIds)
    reward = 0
    state0 = initState

    episodeStates = []

    for h in range(Horizon):
        start = h * numJoints
        end = start + numJoints
        action = episode[start:end]
        action = torch.Tensor(action)
        state1 = getStateFromNN(neuralNet, action, state0)
        episodeStates.append(state1.tolist())
#       reward += getReward(action, jointIds, futureStates[h], neuralNet, initState)
        reward += getReward(action, jointIds, goalCoords, distToGoal, state0, state1)
        state0 = state1
    return reward, episodeStates

def main():
    # Load Neural Network instead of pybullet stuff
    neuralNet, stateLength, actionLength = loadNN()

    initialTime = time.time() # Starting time

    # Random seed every run so the neural net is trained correctly
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
    initState = [
        0.,          0.,          0.,          0.,          0.,          0.,
        0.,          0.,          0.,          0.,          0.689159,   -0.44118994,
        -0.1271375,   0.17627691, -0.67473705, -0.0150754,   0.16944932, -0.81692621,
        -0.01484894,  0.16529569, -0.81707433, -0.10784882,  0.16529569, -0.8143106,
        -0.10785322,  0.07068605, -0.86438095, -0.25777365,  0.06418841, -0.76471571,
        -0.25793238,  0.07710409, -0.67473704, -0.0150754,   0.16944933
    ]
    initCoords = initState[23:26]

    goalCoords = torch.Tensor(goalCoords)
    initState = torch.Tensor(initState)
    initCoords = torch.Tensor(initCoords)

    debug = {
        'goalCoords': goalCoords,
        'initState': initState,
        'initCoords': initCoords
    }

    # Calculate the distance from initial coordinates to goal coordinates
    distToGoal = dist(torch.Tensor(initCoords), torch.Tensor(goalCoords))

    # traj = makeTrajectory(initCoords, goalCoords)
    print("initCoords:\t", initCoords)
    print("initState:\t", initState)
    print("goalCoords:\t", goalCoords)
    # print("traj:\n", traj)
    
    # Constants:
    # Iterations = len(traj) # N - envSteps
    # Epochs = 40 # T - trainSteps
    # Episodes = 200 # G - plans
    # Horizon = 10 # H - horizonLength
    # TopKEps = int(0.3*Episodes)

    # Constants:
    MAX_ITERATIONS = 3 # Program will quit after failing to reach goal for this number of iterations
    # Iterations = len(traj) # N - envSteps
    Iterations = MAX_ITERATIONS # N - envSteps
    Epochs = 20 # T - trainSteps
    Episodes = 1200 # G - plans
    Horizon = 1 # H - horizonLength
    TopKEps = int(0.15*Episodes) # how many episodes to use to calculate the new mean and covariance for the next epoch

    # Getting the joints' lower and upper limits
    jointMins = [-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, 0, -0.04]
    jointMaxes = [np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, 0.04, 0]
    jointMins = jointMins*Horizon
    jointMaxes = jointMaxes*Horizon
    jointMins = torch.Tensor(jointMins)
    jointMaxes = torch.Tensor(jointMaxes)

    # List of final state-action-state pairs
    saveRun = []
    saveAction = []

    # Final end-effector positions
    finalEePos = []

    # Time per iteration taken
    iterationTimes = []

    # Initialize EE starting position
    eePos = initCoords

    # The loop for stepping through each environment steps
    for envStep in range(Iterations):
        startTime = time.time()
        print(f"Running Iteration {envStep} ...")

        # Calculate distance from start to goal
        print("eePos:\t", eePos)
        print("goalCoords:\t", goalCoords)
        distError = dist(eePos, goalCoords)
        print("prevDistToGoal:\t", distError)

        # Initialize mean and covariance
        mu = torch.Tensor([0]*(len(ACTIVE_JOINTS) * Horizon))
        cov = torch.eye(len(mu)) * ((np.pi/2) ** 2)

        # futureStates = []
        # for h in range(Horizon):
        #     if envStep + h > len(traj) - 1:
        #         futureStates.append(traj[-1])
        #     else:
        #         futureStates.append(traj[envStep + h])
        # futureStates = torch.stack(futureStates)

        epsMem = [] # List to store all episodes and their associated costs

        # The loop for improving the distribution
        for e in range(Epochs):
            print(f"Epoch {e}")

            # Initialize the distribution which we sample our actions from
            distr = torch.distributions.MultivariateNormal(mu, cov)

            # The loop for generating all the required episodes
            for eps in range (Episodes):
                episode = distr.sample() # Sample an episode of actions
                
                # Make sure samples don't exceed the joints' limit
                episode = torch.clamp(episode, jointMins, jointMaxes).tolist()
                
                # Calculates the cost of each episode
                cost, episodeStates = getEpsReward(episode, ACTIVE_JOINTS, Horizon, goalCoords, distToGoal, neuralNet, initState)
                # cost = getEpsReward(episode, ACTIVE_JOINTS, Horizon, futureStates, neuralNet, initState)
                
                epsMem.append((episode, cost, episodeStates)) # Save the episodes and their associated costs

            # Sort episodes by cost, ascending
            epsMem = sorted(epsMem, key = lambda x: x[1])

            # Keep the top K episodes
            epsMemTopK = epsMem[0:TopKEps]

            # Remove the cost element from these episodes
            topK = [x[0] for x in epsMemTopK]
            topK = torch.Tensor(topK)

            # Calculate the new mean and covariance
            mu = torch.mean(topK, axis = 0)
            std = torch.std(topK, axis = 0)
            var = torch.square(std)
            noise = torch.Tensor([0.2]*Horizon*len(ACTIVE_JOINTS))
            var = var + noise
            cov = torch.Tensor(np.diag(var))
            currEpsNum = Episodes - TopKEps

        # Extract the best action
        print("epsMem:\n", epsMem[0])
        bestAction = epsMem[0][0][0:len(ACTIVE_JOINTS)]
        eePos = epsMem[0][2][0][23:26]
        saveAction.append(bestAction)

        # Save the time taken by this iteration
        iterationTime = time.time() - startTime
        iterationTimes.append(iterationTime)

        # saveAction.extend(initState[8:-1])
        
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
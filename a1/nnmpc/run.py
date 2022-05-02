from random import getstate
from pygame import init
import torch
import numpy as np
import math
import pickle
import os
import argparse
from torch import nn


# Ideal height for dog to maintain
robotHeight = 0.420393
minState = [-1.02153601, -1.1083865, 0.04107661, -1.04536732, -1.03269277 , 0.02834534, -1.2568753, -0.91284767, 0.06280416, -1.25270735, -0.83711144, 0.06090062, -1.11715088, -0.97780094, 0.06341114]
minAction = [-0.80285144, -1.04719758, -2.69653368, -0.80285144, -1.04719758, -2.69653368, -0.80285144, -1.04719758, -2.69653368, -0.80285144, -1.04719758, -2.69653368]
stateRange = [5.35852929, 2.55355967, 0.5749573, 5.37398911, 2.57010978, 0.59185147, 5.24135348, 2.27949571, 0.50033069, 5.2288108300000005, 2.29599363, 0.51464844, 5.2859171400000005, 2.43478941, 0.40024636]
actionRange = [1.60570288, 5.2359879, 1.78023583, 1.60570288, 5.2359879, 1.78023583, 1.60570288, 5.2359879, 1.78023583, 1.60570288, 5.2359879, 1.78023583]


def loadNN(args):
    '''
    Description:
    Loads the NN from the specified file

    Inputs:
    args: command line arguments

    Returns:
    :neuralNet- {torch.nn.Module} - The neural network
    :stateLength- {int} - The length of the state
    :actionLength- {int} - The length of the action
    '''
    stateLength = 15
    actionLength = 12

    # Load the neural network
    neuralNet = nn.Sequential(
                        nn.Linear(stateLength + actionLength, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Linear(256, stateLength)
                     )
    # Edit this when done with training
    neuralNet.load_state_dict(torch.load(args.model_folder))
    neuralNet.eval()

    return neuralNet, stateLength, actionLength

def normalizeState(state):
    '''
    Description:
    Normalizes the state

    Inputs:
    :state: {list} - The state to be normalized

    Returns:
    :normState: {torch.Tensor} - The normalized state
    '''
    global minState, stateRange
    diff = np.subtract(state, minState)
    # exit()
    normalState = diff/stateRange
    return normalState.tolist()

def normalizeAction(action):
    '''
    Description:
    Normalizes the action

    Inputs:
    :action: {list} - The action to be normalized

    Returns:
    :normAction: {torch.Tensor} - The normalized action
    '''
    global minAction, actionRange
    diff = np.subtract(action, minAction)
    normalAction = diff/actionRange
    return normalAction.tolist()

def unNormalizeState(normState):
    '''
    Description:
    Un-normalizes the state

    Inputs:
    :normState: {torch.Tensor} - The normalized state

    Returns:
    :state: {list} - The un-normalized state
    '''
    normState = normState.detach()
    global minState, stateRange
    prod = np.multiply(normState, stateRange)
    state = np.add(prod, minState)
    return (state.tolist())

def getStateFromNN(neuralNet, action, initialState):
    '''
    Description:
    Predicts the next state given the current state and action

    Inputs:
    :neuralNet: {torch.nn.Module} - The neural network
    :action: {list} - The action to be taken
    :initialState: {list} - The current state

    Returns:
    :nnPredState: {torch.Tensor} - The predicted state
    '''
    # Predict the next state with state1 + action
    state_action = []
    s1 = normalizeState(initialState)
    a = normalizeAction(action)
    state_action.extend(s1)
    state_action.extend(a)
    nnPredState = neuralNet.forward(torch.Tensor(state_action))
    nnPredState = nnPredState.detach()
    s2 = unNormalizeState(nnPredState)
    return s2


def getWeightedState(nnPredState):
    '''
    Description:
    Gets the weighted state

    Inputs:
    :nnPredState: {torch.Tensor} - The predicted state

    Returns:
    :weightedState: {torch.Tensor} - The weighted state
    '''
    # Get hip and floating base from new state
    hips = nnPredState[:12]
    base_pos = nnPredState[12:]

    # Calculate pitch, roll, yaw
    pitchR = abs(hips[2] - hips[8])
    pitchL = abs(hips[5] - hips[11])
    rollF = abs(hips[2] - hips[5])
    rollR = abs(hips[8] - hips[11])
    yawR = abs(hips[1] - hips[7])
    yawL = abs(hips[4] - hips[10])

    # Ideal height for dog to maintain
    global robotHeight
    # Goal point for dog to reach
    goalPoint = [10, 0, robotHeight]
    distance = math.dist(base_pos, goalPoint)**2
    heightErr = abs(robotHeight - base_pos[2])

    state = torch.Tensor([pitchR, pitchL, rollF, rollR, yawR, yawL, distance, heightErr])
    return state


def getReward(state):
    '''
    Description:
    Gets the reward for the given state

    Inputs:
    :state: {list} - The state to get the reward for

    Returns:
    :reward: {float} - The reward
    '''
    w = torch.Tensor([2000,2000,900,900,300,300,2,1000])
    reward = (w*state). sum().numpy()
    if state[-1] > 0.25:
        reward += 1000
    return reward


def getEpsReward(neuralNet, actionLength, initialState, episode, jointIds, Horizon):
    '''
    Description:
    Gets the reward for the whole episode

    Inputs:
    :neuralNet: {torch.nn.Module} - The neural network
    :actionLength: {int} - The length of the action
    :initialState: {list} - The initial state
    :episode: {list} - The episode to get the reward for
    :jointIds: {list} - The joint ids
    :Horizon: {int} - The horizon

    Returns:
    :reward: {float} - The reward
    '''
    numJoints = len(jointIds)
    reward = 0
    startDist, endDist = 0,0
    state0 = initialState
    for h in range(Horizon):
        start = h*numJoints
        end = start + numJoints
        action = episode[start:end]
        state1 = getStateFromNN(neuralNet, action, state0)
        reward += getReward(getWeightedState(state1))
        state0 = state1     # this is to continue where the state left off

        if h == (Horizon-1):
            futureS = start
            futureE = end
            endDist = state1[6]
        else:
            futureS = end
            futureE = end + numJoints

        actionMag = 8 * math.dist(episode[futureS:futureE], action)
        reward += actionMag

        if h == 2:
            startDist = state1[6]

    if startDist < endDist:
        # print(f"START: {startDist}")
        # print(f"END: {endDist}")
        reward += 10000
    # exit()
    return reward

def main(args):
    print("LOADING NN ... ")
    # Load Neural Network instead of pybullet stuff
    neuralNet, stateLength, actionLength = loadNN(args)
    print("FINISEHD LOADING NN!")

    # Hard code data without pybullet
    jointIds = []
    jointMins = []
    jointMaxes = []
    initialState = []

    jointIds = [2,3,4,6,7,8,10,11,12,14,15,16]
    jointMins = [-0.802851455917, -1.0471975512, -2.69653369433, -0.802851455917, -1.0471975512, -2.69653369433, -0.802851455917, -1.0471975512, -2.69653369433, -0.802851455917, -1.0471975512, -2.69653369433]
    jointMaxes = [0.802851455917, 4.18879020479, -0.916297857297, 0.802851455917, 4.18879020479, -0.916297857297, 0.802851455917, 4.18879020479, -0.916297857297, 0.802851455917, 4.18879020479, -0.916297857297]
    initialState = [
        0.179689, -0.047635, 0.42,   # FR_hip_joint
        0.179689, 0.047635, 0.42,    # FL_hip_joint
        -0.179689, -0.047635, 0.42,     # RR_hip_joint
        -0.179689, 0.047635, 0.42,      # RL_hip_joint
        0.012731, 0.002186, 0.42     # Floating base
    ]

    # Initialize variables
    Iterations = 10
    Epochs = 3
    Episodes = 70
    Horizon = 80
    TopKEps = int(0.2*Episodes)
    numJoints = len(jointIds)
    jointMins = jointMins*Horizon
    jointMaxes = jointMaxes*Horizon
    jointMins = torch.Tensor(jointMins)
    jointMaxes = torch.Tensor(jointMaxes)
    bestActions = []
    centerTraj = []

    print("RUNNING MPC ...")
    # MPC
    for iter in range(Iterations):
        print(f"Running Iteration {iter} ...")
        # print(initialState)
        mu = torch.Tensor([0]*(numJoints * Horizon))
        cov = torch.eye(len(mu)) * ((np.pi/2) ** 2)
        # this is what we should be resetting to
        # number of episodes to sample
        currEpsNum = Episodes
        # This is the "memory bank" of episodes we are going to use
        epsMem = []
        for e in range(Epochs):
            print(f"Epoch {e}")
            # print(initialState)
            # initialize multivariate distribution
            distr = torch.distributions.MultivariateNormal(mu, cov)
            # Now we get the episodes
            for eps in range(currEpsNum):
                # generate episode
                episode = distr.sample()
                # make sure it's valid by clamping with the mins and maxes
                episode = torch.clamp(episode, jointMins, jointMaxes).tolist()
                # get cost of episode
                cost = getEpsReward(neuralNet, actionLength, initialState, episode, jointIds, Horizon)
                # store the episode, along with the cost, in episode memory
                epsMem.append((episode,cost))

            # Sort the episode memory
            epsMem = sorted(epsMem, key = lambda x: x[1])

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
        # print("Reward Taken: ", epsMem[0][1])
        # Save best action and state
        bestActions.extend(epsMem[0][0][0:numJoints])
        # Set the new state
        initialState = getStateFromNN(neuralNet, epsMem[0][0][0:numJoints], initialState)
        print(f"AT ITERATION {iter} WE HAVE CENTER AT ({initialState[12:]})")
        centerTraj.append(initialState[12:])

    folder = f"./NN_MPC_Action_Results/"
    if not os.path.exists(folder):
        # create directory if not exist
        os.makedirs(folder)

    print("DONE!!!!!")
    with open(folder + f"V1_run_I{Iterations}_E{Epochs}_Eps{Episodes}.pkl", 'wb') as f:
        pickle.dump(bestActions, f)
    
    trajFolder = f"./trajectories/"
    if not os.path.exists(trajFolder):
        # create directory if not exist
        os.makedirs(trajFolder)

    with open(trajFolder + f"V1_run_I{Iterations}_E{Epochs}_Eps{Episodes}_NNPREDICTED.pkl", 'wb') as f:
        pickle.dump(centerTraj, f)



if __name__ == '__main__':
    modelFolder = "mult_models/V1_Model/V1_Model_model1.pt"

    parser = argparse.ArgumentParser(description='Running MPC with NN instead of Pybullet')
    parser.add_argument('--model-folder', type=str, default=modelFolder, help="path to model")
    args = parser.parse_args()
    
    main(args)

import torch
import numpy as np
import math
import pickle
import os
import argparse
from torch import nn


# Ideal height for dog to maintain
robotHeight = 0.420393

def loadNN(args):
    stateLength = 0
    actionLength = 0
    # Edit for size of input
    if args.robot == "a1":
        stateLength = 15
        actionLength = 12
    else:   # UR5
        stateLength = 3
        actionLength = 8

    # Load the neural network
    neuralNet = nn.Sequential(
                    nn.Linear(stateLength + actionLength, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
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


def getStateFromNN(neuralNet, action, initialState):
    # Predict the next state with state1 + action
    state_action = []
    state_action.extend(initialState)
    state_action.extend(action)
    nnPredState = neuralNet.forward(torch.Tensor(state_action))
    return nnPredState


def getWeightedState(nnPredState):
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
    # np_seed = np.random.randint(low=0, high=1000)
    # np.random.seed(np_seed)
    # goal_x = np.random.uniform(low=-10, high=10)
    # goal_y = np.random.uniform(low=-10, high=10)
    # Goal point for dog to reach
    goalPoint = [10, 0, robotHeight]
    distance = math.dist(base_pos, goalPoint)**2
    heightErr = abs(robotHeight - base_pos[2])

    state = torch.Tensor([pitchR, pitchL, rollF, rollR, yawR, yawL, distance, heightErr])
    return state


def getReward(state):
    w = torch.Tensor([10000, 10000,10000,10000,900,900,2,3000])
    reward = (w*state).sum().numpy()
    if state[-1] > 0.25:
        reward += 1000
    return reward


def getEpsReward(neuralNet, actionLength, initialState, episode, jointIds, Horizon):
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
            endDist = state1.tolist()[6]
        else:
            futureS = end
            futureE = end + numJoints

        actionMag = 8 * math.dist(episode[futureS:futureE], action)
        reward += actionMag

        if h == 2:
            startDist = state1.tolist()[6]

    if startDist < endDist:
        # print(f"START: {startDist}")
        # print(f"END: {endDist}")
        reward += 10000
    # exit()
    return reward

def main(args):
    # Load Neural Network instead of pybullet stuff
    neuralNet, stateLength, actionLength = loadNN(args)

    # Hard code data without pybullet
    jointIds = []
    jointMins = []
    jointMaxes = []
    initialState = []
    if args.robot == "a1":
        jointIds = [2,3,4,6,7,8,10,11,12,14,15,16]
        jointMins = [-0.802851455917, -1.0471975512, -2.69653369433, -0.802851455917, -1.0471975512, -2.69653369433, -0.802851455917, -1.0471975512, -2.69653369433, -0.802851455917, -1.0471975512, -2.69653369433]
        jointMaxes = [0.802851455917, 4.18879020479, -0.916297857297, 0.802851455917, 4.18879020479, -0.916297857297, 0.802851455917, 4.18879020479, -0.916297857297, 0.802851455917, 4.18879020479, -0.916297857297]
        initialState = [
            0.179689, -0.047635, 0.42040155674978175,   # FR_hip_joint
            0.179689, 0.047635, 0.42040155674978175,    # FL_hip_joint
            -0.179689, -0.047635, 0.42040155674978175,     # RR_hip_joint
            -0.179689, 0.047635, 0.42040155674978175,      # RL_hip_joint
            0.012731, 0.002186, 0.42040155674978175     # Floating base
        ]
    else:   # ur5
        jointIds = [1,2,3,4,5,6,8,9]
        jointMins = []
        jointMaxes = []
        initialState = []
        pass

    # Initialize variables
    Iterations = 150
    Epochs = 10
    Episodes = 100
    Horizon = 100
    TopKEps = int(0.3*Episodes)
    numJoints = len(jointIds)
    jointMins = jointMins*Horizon
    jointMaxes = jointMaxes*Horizon
    jointMins = torch.Tensor(jointMins)
    jointMaxes = torch.Tensor(jointMaxes)
    bestActions = []

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
        initialState = getStateFromNN(neuralNet, epsMem[0][0][0:numJoints], initialState).tolist()

    folder = f"./testNNmpc/"
    if not os.path.exists(folder):
        # create directory if not exist
        os.makedirs(folder)

    print("DONE!!!!!")
    with open(folder + f"A1_run_I{Iterations}_E{Epochs}_Eps{Episodes}_H{Horizon}_model_A1_model_gb_layers3_path69_epoch3.pkl", 'wb') as f:
        pickle.dump(bestActions, f)


if __name__ == '__main__':
    modelFolder = "./models/A1_model_gb_layers3_path69_epoch3.pt"

    parser = argparse.ArgumentParser(description='Training a Neural Network with the Best Actions')
    parser.add_argument('--model-folder', type=str, default=modelFolder, help="path to model")
    parser.add_argument('--robot', type=str, default="a1", help='robot type')
    args = parser.parse_args()
    
    main(args)

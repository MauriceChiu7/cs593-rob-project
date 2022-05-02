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
minState = [-1.0297287398795225, -1.6644083654026904, -0.015584937923837375, -1.0021420251033062, -1.6705273018242133, -0.008343590749213488, -0.8590960085231067, -1.7285345399750789, -0.02795621764922601, -1.0578811858705537, -1.641913003325433, -0.0025479132536383314, -1.068227497165033, -1.6219328574424108, -0.022971474016212648, -0.8770603655513525, -1.6987640819269179, -0.039889400106911624, -1.2568752990223655, -1.3210953331392936, 0.011956009963431019, -1.2557680387705799, -1.3214811303046041, 0.005454480898304741, -1.2829497394833986, -1.3814368399802714, -0.04579027727923243, -1.2527073515093285, -1.298596686420213, 0.017226176287621603, -1.2565594381598746, -1.284249218438306, -0.011482361365547256, -1.3007383470144487, -1.418266284133236, -0.03336962572266933, -1.1171508752374961, -1.4930602742729848, 0.03155428938255963]
minAction = [-0.80285144, -1.04719758, -2.69653368, -0.80285144, -1.04719758, -2.69653368, -0.80285144, -1.04719758, -2.69653368, -0.80285144, -1.04719758, -2.69653368]
stateRange = [4.761497558240923, 3.7112114299483405, 0.7519433260297952, 4.776578272504329, 3.712938386985593, 0.7578246885761932, 4.775229545743272, 3.9290905825926625, 0.8850212907837247, 4.731175528692843, 3.7108143173557067, 0.699352191476181, 4.710560939573208, 3.704278615778671, 0.7299159716380068, 4.638382378077592, 3.8709040566524915, 0.8448035247828886, 4.858898273711912, 3.050115272740485, 0.6355866880288158, 4.918093853280212, 3.0308168974654284, 0.6695222803466775, 5.033055561619363, 3.232053531269027, 0.7371904560779658, 4.762149383040643, 3.0497447957559833, 0.6084018678140535, 4.707957095207612, 3.046731769249619, 0.6718921218839282, 4.844659306880555, 3.3087431860563856, 0.8282433731473829, 4.72570788735529, 3.4035906171442045, 0.5369954942580496]
actionRange = [1.60570288, 5.2359879, 1.78023583, 1.60570288, 5.2359879, 1.78023583, 1.60570288, 5.2359879, 1.78023583, 1.60570288, 5.2359879, 1.78023583]


def loadNN(args):
    stateLength = 39
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
    global minState, stateRange
    diff = np.subtract(state, minState)
    # exit()
    normalState = diff/stateRange
    return normalState.tolist()

def normalizeAction(action):
    global minAction, actionRange
    diff = np.subtract(action, minAction)
    normalAction = diff/actionRange
    return normalAction.tolist()

def unNormalizeState(normState):
    normState = normState.detach()
    global minState, stateRange
    prod = np.multiply(normState, stateRange)
    state = np.add(prod, minState)
    return (state.tolist())

def getStateFromNN(neuralNet, action, initialState):
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
    # Get hip and floating base from new state
    hips = []
    hips.extend(nnPredState[:3])
    hips.extend(nnPredState[9:12])
    hips.extend(nnPredState[18:21])
    hips.extend(nnPredState[30:33])
    # print(len(hips))
    # exit(0)
    # hips = nnPredState[:12]
    base_pos = nnPredState[36:]
    # print(base_pos)

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
    w = torch.Tensor([2000,2000,900,900,300,300,2,1000])
    reward = (w*state). sum().numpy()
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
    # initialState = [
    #     0.179689, -0.047635, 0.42,   # FR_hip_joint
    #     0.179689, 0.047635, 0.42,    # FL_hip_joint
    #     -0.179689, -0.047635, 0.42,     # RR_hip_joint
    #     -0.179689, 0.047635, 0.42,      # RL_hip_joint
    #     0.012731, 0.002186, 0.42     # Floating base
    # ]
    initialState = [0.17149816867732603, -0.0472022998918858, 0.4198066807921248, 0.17159858284039164, -0.10931785381849503, 0.39250901096377894, 0.18532143380864324, -0.13192638861245765, 0.11277740016922333, 0.17149437688451, 0.04806755015342688, 0.41997334043764883, 0.1716063709987689, 0.11012719244757097, 0.3925468384555968, 0.18538846251086838, 0.13216509389547826, 0.1127674872502053, -0.1878798274402711, -0.0472166944583873, 0.4198575069536097, -0.19413313438307614, -0.10933068723867263, 0.3925263704894463, -0.1796674996375291, -0.13192034079748197, 0.11274725783740674, -0.18788361923175695, 0.048053159272305655, 0.4200241770389607, -0.19413229590075556, 0.11011111348497499, 0.39256614216389485, -0.17962935251684564, 0.13213469968925276, 0.11274171680685058, 0.004538256143277372, 0.00261098341925018, 0.42040155674978175]

    # Initialize variables
    Iterations = 100
    Epochs = 25
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

    folder = f"./NEWNN_MPC_Action_Results/"
    if not os.path.exists(folder):
        # create directory if not exist
        os.makedirs(folder)

    print("DONE!!!!!")
    with open(folder + f"V1_run_I{Iterations}_E{Epochs}_Eps{Episodes}.pkl", 'wb') as f:
        pickle.dump(bestActions, f)
    
    trajFolder = f"./NEWtrajectories/"
    if not os.path.exists(trajFolder):
        # create directory if not exist
        os.makedirs(trajFolder)

    with open(trajFolder + f"V1_run_I{Iterations}_E{Epochs}_Eps{Episodes}_NNPREDICTED.pkl", 'wb') as f:
        pickle.dump(centerTraj, f)



if __name__ == '__main__':
    modelFolder = "./NEWmult_models/V1_Model/V1_Model_model1.pt"

    parser = argparse.ArgumentParser(description='Running MPC with NN instead of Pybullet')
    parser.add_argument('--model-folder', type=str, default=modelFolder, help="path to model")
    args = parser.parse_args()
    
    main(args)

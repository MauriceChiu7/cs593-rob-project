import argparse
from operator import le
import os
import torch
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from torch import nn


# Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self, stateLength, actionLength):
        super(NeuralNetwork, self).__init__()

        self.fc = nn.Sequential(
                        nn.Linear(stateLength + actionLength, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Linear(256, stateLength),
                     )

    def forward(self, x):
        x = self.fc(x)
        return x


MIN_STATES = [
    -3.10712409, -3.06124139, -2.86276412, -4.19966221, -3.44298506, -3.12980533,
    -1.45498884, -0.92632586,  0.,          0.,          0.689159,   -0.45491707,
    -0.4585793,   0.09143765, -0.673587,   -0.67331188,  0.05082713, -0.81560997,
    -0.81535459,  0.03439972, -0.8188093,  -0.82121336,  0.03439972, -0.90763709,
    -0.88345045,  0.07974603, -1.38587203, -0.97608102, -0.09001164, -1.77162679,
    -1.70116438, -0.29742956, -0.67358702, -0.67331189,  0.05082713, -3.14159274,
    -3.14159274, -3.14159274, -3.14159274, -3.14159274, -3.14159274, -0.,
    -0.04,       -3.14183736, -3.14154696, -2.92847085, -3.14190817, -3.1661272,
    -3.16744399, -1.1899116,  -0.81518328,  0.,          0.,          0.689159,
    -0.41903543, -0.42967215,  0.09143765, -0.65547734, -0.57306675,  0.12626202,
    -0.79439353, -0.67791404,  0.14837159, -0.77439313, -0.71497727,  0.14837159,
    -0.69320744, -0.71282327,  0.12966966, -1.38587203, -0.82441626, -0.09001164,
    -1.00950668, -1.31728623,  0.02539214, -0.65547734, -0.57306677,  0.12626202,
    ]
MAX_STATES = [
    3.75653076, 0.55526543, 2.80523705, 3.1437099,  3.19662118, 3.30458975,
    0.81385142, 1.92828441, 0.00000001, 0.00000001, 0.68915901, 0.45862067,
    0.45844874, 0.61817586, 0.65903576, 0.67313442, 0.84359277, 0.79452851,
    0.81517503, 0.97899382, 0.81766808, 0.81246545, 0.97899382, 0.87753732,
    0.89807087, 0.98115864, 1.28755629, 0.98191874, 1.01833462, 0.92718923,
    1.01470949, 0.99016566, 0.65903574, 0.67313445, 0.84359276, 3.14159274,
    3.14159274, 3.14159274, 3.14159274, 3.14159274, 3.14159274, 0.04,
    0.,         3.75653076, 0.55526543, 2.61296391, 3.17274809, 3.19662118,
    3.30458975, 0.60485941, 0.87994975, 0.00000001, 0.00000001, 0.68915901,
    0.45165157, 0.43449998, 0.56384039, 0.65903576, 0.64585759, 0.72415397,
    0.79452851, 0.77185048, 0.78790315, 0.80996472, 0.77765211, 0.78790315,
    0.8579651,  0.8281822,  0.7657455,  0.73982674, 0.87553448, 0.75971469,
    0.76232534, 0.86133107, 0.75713578, 0.65903574, 0.64585757, 0.724154,  
    ]

# MIN_STATES = [-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, 0, -0.04]
# MAX_STATES = [np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, 0.04, 0]
stateRange = np.subtract(MAX_STATES, MIN_STATES)

# print(stateRange)
# exit()
# def normalizeData(data):
#     normalizedData = []
#     for d in data:
#         prevState = d[0:8]
#         prevEEPos = d[8:11]
#         action = d[11:19]
#         nextState = d[19:27]
#         nextEEPos = d[27:30]

#         normalizedState = []
#         normalizedState.extend(normalize(prevState))
#         normalizedState.extend(prevEEPos)
#         normalizedState.extend(normalize(action))
#         normalizedState.extend(normalize(nextState))
#         normalizedState.extend(nextEEPos)

#         normalizedData.append(normalizedState)

#     return normalizedData


def normalize(data):
    diff = np.subtract(data, MIN_STATES)
    normalState = diff/stateRange
    # print(normalState)
    # exit()
    return normalState

def unnormalize(normalizedData):
    return np.add(normalizedData * stateRange, MIN_STATES)

# TODO: normalize inputs and unnormalize output
def train(args):
    print(f"Training...from {args.training_folder}")

    LAYERS = 2
    EPOCH = 50
    modelName = f"UR5_V1_Model_{LAYERS}Layers"
    # create model folder
    modelFolder = f"./mult_models/{modelName}/"
    if not os.path.exists(modelFolder):
        # create directory if not exist
        os.makedirs(modelFolder)

    stateLength = 35
    actionLength = 8

    
    # Open pkl file
    tups = None
    allData = []

    # Read through trainingData
    _, _, files = next(os.walk(args.training_folder))
    for fi in files:
        # Open every file and combine the state_action pairs
        currName = args.training_folder + fi
        with open(currName, 'rb') as f:
            tups = pickle.load(f)

        tups = np.array(tups)
        tups = normalize(tups)

        # Compile all data
        allData.extend(tups)


    random.shuffle(allData)
    trainRatio = 0.9
    length = int(len(allData)*trainRatio)
    trainData = allData[:length]
    testData = allData[length:]


    # Initialize neural network
    neuralNet = NeuralNetwork(stateLength, actionLength)
    optimizer = torch.optim.Adam(neuralNet.parameters(), lr=0.01)

    # Record MSE for each epoch
    mse = nn.MSELoss()

    trainMSE = []
    testMSE = []
    for e in range(EPOCH):
        print("Epoch: ", e)
        currTrainMSE = []
        # Shuffle state-action pairs after each epoch
        random.shuffle(trainData)
        random.shuffle(testData)

        for t in allData:
            state_action = t[:stateLength+actionLength]
            nnTarget = t[stateLength+actionLength:]

            # Pass into Neural Net and get MSE Loss
            nnPred = neuralNet.forward(torch.Tensor(state_action))
            loss = mse(nnPred, torch.Tensor(nnTarget))
            currTrainMSE.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avgTrainMSE = np.mean(currTrainMSE)
        trainMSE.append(avgTrainMSE)
        print("Training Loss: ", avgTrainMSE)

        torch.save(neuralNet.fc.state_dict(), modelFolder + f"{modelName}_epoch{e}.pt")

        # Testing
        currTestMSE = []
        
        for t in testData:
            state_action = t[:stateLength+actionLength]
            nnTarget = t[stateLength+actionLength:]

            # Pass into Neural Net and get MSE Loss
            nnPred = neuralNet.forward(torch.Tensor(state_action))
            loss = mse(nnPred, torch.Tensor(nnTarget))
            currTestMSE.append(loss.item())
        avgTestMSE = np.mean(currTestMSE)
        testMSE.append(avgTestMSE)
        print("Testing Loss: ", avgTestMSE)
    

    torch.save(neuralNet.fc.state_dict(), modelFolder + f"{modelName}_model1.pt")
    print("DONE!")


    t = f"{modelName}: Training and Testing Average MSE Across Epochs"
    
    # Plot graph
    plt.xlabel("Epochs")
    plt.ylabel("Average MSE")
    plt.plot(trainMSE, label = "Training MSE")
    plt.plot(testMSE, label = "Testing MSE")
    plt.title(t)
    plt.legend()
    

    graphFolder = "./graphs/"
    if not os.path.exists(graphFolder):
        # create directory if not exist
        os.makedirs(graphFolder)
    plt.savefig(f"{graphFolder}{modelName}_MSE_Results.png")

    plt.show()


if __name__ == '__main__':
    trainingFolder = f"./trainingData/"
    testingFolder = f"./testData/"

    parser = argparse.ArgumentParser(description='Training a Neural Network with the Best Actions')

    parser.add_argument('--testing-folder', type=str, default=testingFolder, help='path to training folder')
    parser.add_argument('--training-folder', type=str, default=trainingFolder, help='path to training folder')
    args = parser.parse_args()

    train(args)
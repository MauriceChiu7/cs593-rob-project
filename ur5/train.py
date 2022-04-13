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
    -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, 0, -0.04, -0.9208793640136719, -0.9239162802696228, -0.7005515694618225, 
    -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, 0, -0.04, 
    -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, 0, -0.04, -0.9208793640136719, -0.9239162802696228, -0.7005515694618225
    ]
MAX_STATES = [
    np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, 0.04, 0, 0.9053947925567627, 0.9046874642372131, 1.1148362159729004, 
    np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, 0.04, 0, 
    np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, 0.04, 0, 0.9053947925567627, 0.9046874642372131, 1.1148362159729004]
# MIN_STATES = [-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, 0, -0.04]
# MAX_STATES = [np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, 0.04, 0]
stateRange = np.subtract(MAX_STATES, MIN_STATES)
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

    stateLength = 11
    actionLength = 8

    
    # Open pkl file
    tups = None
    allData = []

    # Read through trainingDataWithEE
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
    trainingFolder = f"./trainingDataWithEE/"
    testingFolder = f"./testData/"

    parser = argparse.ArgumentParser(description='Training a Neural Network with the Best Actions')

    parser.add_argument('--testing-folder', type=str, default=testingFolder, help='path to training folder')
    parser.add_argument('--training-folder', type=str, default=trainingFolder, help='path to training folder')
    args = parser.parse_args()

    train(args)
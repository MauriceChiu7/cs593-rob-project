import argparse
import os
import torch
import pickle
import random
import matplotlib.pyplot as plt
from torch import nn


# Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self, stateLength, actionLength):
        super(NeuralNetwork, self).__init__()

        self.fc = nn.Sequential(
                        nn.Linear(stateLength + actionLength, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, stateLength),
                     )

    def forward(self, x):
        x = self.fc(x)
        return x


# TODO: normalize inputs and outputs
def train(args):
    print("Training...")
    # Edit for size of input
    if args.robot == "a1":
        stateLength = 15
        actionLength = 12
    else:   # UR5
        stateLength = 3
        actionLength = 8
    
    # Open pkl file
    tups = None
    allData = []

    # Read through training directory
    _, _, files = next(os.walk(args.training_folder))
    for fi in files:
        # Open every file and combine the state_action pairs
        currName = args.training_folder + fi
        with open(currName, 'rb') as f:
            tups = pickle.load(f)

        # Need to update the state1's and get rid of the last pair bc nothing to compare with
        for i in range(len(tups)-1):
            tups[i][-stateLength:] = tups[i+1][:stateLength]
        tups = tups[:-1]

        allData.extend(tups)

    # Shuffle state-action pairs
    random.shuffle(allData)

    # Initialize neural network
    neuralNet = NeuralNetwork(stateLength, actionLength)
    optimizer = torch.optim.Adam(neuralNet.parameters(), lr=0.01)

    mse = nn.MSELoss()
    for t in allData:
        state_action = t[:stateLength+actionLength]
        nnTarget = t[stateLength+actionLength:]

        # Pass into Neural Net and get MSE Loss
        nnPred = neuralNet.forward(torch.Tensor(state_action))
        loss = mse(nnPred, torch.Tensor(nnTarget))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Loss: ", loss)

    # Save model
    modelFolder = "./models/"
    if not os.path.exists(modelFolder):
        # create directory if not exist
        os.makedirs(modelFolder)
    
    if args.robot == "a1":
        torch.save(neuralNet.fc.state_dict(), modelFolder + "A1_model_3.pt")
    else:   # UR5
        torch.save(neuralNet.fc.state_dict(), modelFolder + "UR5_model.pt")


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
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, stateLength),
                )
    # Edit this when done with training
    neuralNet.load_state_dict(torch.load(args.model_folder))
    neuralNet.eval()

    return neuralNet, stateLength, actionLength


def test(args):
    print("Testing...")
    # Load Neural Network model to test
    neuralNet, stateLength, actionLength = loadNN(args)

    # Open testData
    tups = None
    allData = []

    allMSE = []

    # Read through test directory
    _, _, files = next(os.walk(args.testing_folder))
    for fi in files:
        # Open every file and combine the state_action pairs
        print(fi)
        currName = args.testing_folder + fi
        with open(currName, 'rb') as f:
            tups = pickle.load(f)

        # Need to update the state1's and get rid of the last pair bc nothing to compare with
        for i in range(len(tups)-1):
            tups[i][-stateLength:] = tups[i+1][:stateLength]
        tups = tups[:-1]


        mse = nn.MSELoss()
        for t in tups:
            state_action = t[:stateLength+actionLength]
            nnTarget = t[stateLength+actionLength:]

            # Pass into Neural Net and get MSE Loss
            nnPred = neuralNet.forward(torch.Tensor(state_action))
            loss = mse(nnPred, torch.Tensor(nnTarget))
            allMSE.append(loss.item())

    # print(len(allMSE))
    t = f"A1: MSE per State Predicton"
    
    # Plot graph
    plt.xlabel("State prediction")
    plt.ylabel("MSE")

    plt.plot(allMSE, marker='o')
    plt.title(t)
    plt.show()
  

if __name__ == '__main__':
    Iterations = 300
    Epochs = 10
    Episodes = 100
    Horizon = 50

    modelFolder = "./models/A1_model_2.pt"
    trainingFolder = f"./unitree_pybullet/trainingData/iter_{Iterations}_epochs_{Epochs}_episodes_{Episodes}_horizon_{Horizon}/"
    testingFolder = f"./unitree_pybullet/testData/iter_{Iterations}_epochs_{Epochs}_episodes_{Episodes}_horizon_{Horizon}/"

    parser = argparse.ArgumentParser(description='Training a Neural Network with the Best Actions')
    parser.add_argument('--mode', type=str, default="test", help='test or train')
    parser.add_argument('--robot', type=str, help='robot type')

    parser.add_argument('--model-folder', type=str, default=modelFolder, help="path to model")
    parser.add_argument('--testing-folder', type=str, default=testingFolder, help='path to training folder')
    parser.add_argument('--training-folder', type=str, default=trainingFolder, help='path to training folder')
    args = parser.parse_args()
    
    if args.mode == "test":
        test(args)
    else:
        train(args)

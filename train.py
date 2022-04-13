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
                        nn.Linear(256, 256),
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


# TODO: normalize inputs and outputs
def train(args):
    print(f"Training...from {args.training_folder}")
    # Edit for size of input
    if args.robot == "a1":
        stateLength = 15
        actionLength = 12
    else:   # UR5
        stateLength = 3
        actionLength = 8

    LAYERS = 3
    EPOCH = 3
    
    # Open pkl file
    tups = None
    allData = []

    # Read through trainingData
    _, _, gfiles = next(os.walk(args.training_folder + "good/"))
    TRAINDATA1 = len(gfiles)
    print("Total Data good files: ", TRAINDATA1)
    for fi in gfiles:
        # Open every file and combine the state_action pairs
        currName = args.training_folder + "good/" + fi
        with open(currName, 'rb') as f:
            tups = pickle.load(f)

        # Need to update the state1's and get rid of the last pair bc nothing to compare with
        for i in range(len(tups)-1):
            tups[i][-stateLength:] = tups[i+1][:stateLength]
        tups = tups[:-1]

        allData.extend(tups)
    
    _, _, bfiles = next(os.walk(args.training_folder + "bad/"))
    TRAINDATA2 = len(bfiles)
    print("Total Data bad files: ", TRAINDATA2)
    for fi in bfiles:
        # Open every file and combine the state_action pairs
        currName = args.training_folder + "bad/" + fi
        with open(currName, 'rb') as f:
            tups = pickle.load(f)

        # Need to update the state1's and get rid of the last pair bc nothing to compare with
        for i in range(len(tups)-1):
            tups[i][-stateLength:] = tups[i+1][:stateLength]
        tups = tups[:-1]

        allData.extend(tups)

    # Read through newtrainingData
    allTest = []
    testgFolder = "./unitree_pybullet/newtrainingData/good/"
    _, _, gfiles = next(os.walk(testgFolder))
    TESTDATA1 = len(gfiles)
    print("Total Test files: ", TESTDATA1)
    for fi in gfiles:
        # Open every file and combine the state_action pairs
        currName = testgFolder + fi
        with open(currName, 'rb') as f:
            tups = pickle.load(f)

        allTest.extend(tups)

    testbFolder = "./unitree_pybullet/newtrainingData/bad/"
    _, _, bfiles = next(os.walk(testbFolder))
    TESTDATA2 = len(bfiles)
    print("Total Test files: ", TESTDATA2)
    for fi in bfiles:
        # Open every file and combine the state_action pairs
        currName = testbFolder + fi
        with open(currName, 'rb') as f:
            tups = pickle.load(f)

        allTest.extend(tups)

    # Initialize neural network
    neuralNet = NeuralNetwork(stateLength, actionLength)
    optimizer = torch.optim.Adam(neuralNet.parameters(), lr=0.01)

    # Record MSE for each epoch
    mseAllEpochs = []
    mseValid = []
    for e in range(EPOCH):
        print("Epoch: ", e)
        # Shuffle state-action pairs after each epoch
        random.shuffle(allData)

        m = []
        mse = nn.MSELoss()
        for t in allData:
            state_action = t[:stateLength+actionLength]
            nnTarget = t[stateLength+actionLength:]

            # Pass into Neural Net and get MSE Loss
            nnPred = neuralNet.forward(torch.Tensor(state_action))
            loss = mse(nnPred, torch.Tensor(nnTarget))
            m.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("Loss: ", loss)
        mseAllEpochs.append(m)

        random.shuffle(allTest)
        # Testing
        m = []
        mse = nn.MSELoss()
        for t in allTest:
            state_action = t[:stateLength+actionLength]
            nnTarget = t[stateLength+actionLength:]

            # Pass into Neural Net and get MSE Loss
            nnPred = neuralNet.forward(torch.Tensor(state_action))
            loss = mse(nnPred, torch.Tensor(nnTarget))
            m.append(loss.item())
        mseValid.append(m)
    
    # Take mean
    mseAllEpochs = torch.mean(torch.Tensor(mseAllEpochs), axis=-1)
    mseValid = torch.mean(torch.Tensor(mseValid), axis=-1)


    # SAVE DATA
    spec = f"A1_model_gb_layers{LAYERS}_path{TRAINDATA1+TRAINDATA2}_epoch{EPOCH}"
    # Save model
    modelFolder = "./models/"
    if not os.path.exists(modelFolder):
        # create directory if not exist
        os.makedirs(modelFolder)
    
    if args.robot == "a1":
        torch.save(neuralNet.fc.state_dict(), modelFolder + spec + ".pt")
    else:   # UR5
        torch.save(neuralNet.fc.state_dict(), modelFolder + spec + ".pt")


    # Save training MSE plot
    plotFolder = "./unitree_pybullet/trainingPlots/"
    if not os.path.exists(plotFolder):
        # create directory if not exist
        os.makedirs(plotFolder)

    t = f"A1: Layers:{LAYERS}, Epochs={EPOCH}, TrainedPaths={TRAINDATA1+TRAINDATA2}, TestedPaths={TESTDATA1+TESTDATA2}"
    
    # Plot graph: training MSE and tested MSE over epochs
    plt.xlabel("Epoch")
    plt.ylabel("MSE")

    plt.plot(mseAllEpochs, marker='o', label="Training Loss")
    plt.plot(mseValid, marker='o', label="Validation Loss")
    plt.legend()
    plt.legend
    plt.title(t)
    plt.savefig(plotFolder + spec + ".png")
    


if __name__ == '__main__':
    # modelFolder = "./models/A1_model_3.pt"
    trainingFolder = f"./unitree_pybullet/trainingData/"
    testingFolder = f"./unitree_pybullet/testData/"

    parser = argparse.ArgumentParser(description='Training a Neural Network with the Best Actions')
    parser.add_argument('--mode', type=str, default="train", help='test or train')
    parser.add_argument('--robot', type=str, default="a1", help='robot type')

    # parser.add_argument('--model-folder', type=str, default=modelFolder, help="path to model")
    parser.add_argument('--testing-folder', type=str, default=testingFolder, help='path to training folder')
    parser.add_argument('--training-folder', type=str, default=trainingFolder, help='path to training folder')
    args = parser.parse_args()
    
    if args.mode == "train":
        train(args)


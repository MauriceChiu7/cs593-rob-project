import argparse
import os
import torch
import pickle
import random
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
    # Open pkl file
    trainingFolder = args.folder
    tups = None
    allData = []

    # Read through training directory
    path, dirs, files = next(os.walk(trainingFolder))
    for x in range(len(files)):
        fname = "sample_{}.pkl".format(x)
        currName = trainingFolder + fname

        with open(currName, 'rb') as f:
            tups = pickle.load(f)
            
        allData.extend(tups)
    
    # Shuffle state-action pairs
    random.shuffle(allData)

    # Edit for size of input
    if args.robot == "a1":
        stateLength = 15
        actionLength = 12
    else:   # UR5
        stateLength = 3
        actionLength = 8

    # Initialize neural network
    neuralNet = NeuralNetwork(stateLength, actionLength)
    optimizer = torch.optim.Adam(neuralNet.parameters(), lr=0.01)

    mse = nn.MSELoss()
    for t in allData:
        state1 = t[:stateLength+actionLength]
        nnTarget = t[stateLength+actionLength:]

        # Pass into Neural Net and get MSE Loss
        nnPred = neuralNet.forward(torch.Tensor(state1))
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
        torch.save(neuralNet.fc.state_dict(), modelFolder + "A1_model.pt")
    else:   # UR5
        torch.save(neuralNet.fc.state_dict(), modelFolder + "UR5_model.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a Neural Network with the Best Actions')
    parser.add_argument('--folder', type=str, default="./training/", help='path to training folder')
    parser.add_argument('--robot', type=str, help='robot type')
    args = parser.parse_args()
    
    train(args)
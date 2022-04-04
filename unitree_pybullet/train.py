import argparse
import pybullet as p
import os
import pybullet_data
import numpy as np
import torch
import csv
import math
import time
import copy
import pickle
from torch import nn


# Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self, input_feats):
        super(NeuralNetwork, self).__init__()

        self.fc = nn.Sequential(
                        nn.Linear(input_feats, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 15),
                     )

    def forward(self, x):
        x = self.fc(x)
        return x


def main(args):
    # Open pkl file
    filename = args.file
    tups = None
    with open(filename, 'rb') as f:
        tups = pickle.load(f)
    

    # edit for size of input
    if args.robot == "a1":
        input_feats = 27
    else:
        input_feats = 3

    # Initialize neural network
    neuralNet = NeuralNetwork(input_feats)
    optimizer = torch.optim.Adam(neuralNet.parameters(), lr=0.01)

    # Train for 
    mse = nn.MSELoss()
    for t in tups:
        state1 = t[:15]
        action1 = t[15:27]
        nnTarget = t[27:]
        state1.extend(action1)

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
    torch.save(neuralNet.fc.state_dict(), modelFolder +"model.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a Neural Network with the Best Actions')
    parser.add_argument('--file', type=str, default=[], help='path file')
    parser.add_argument('--robot', type=str, help='robot type')
    args = parser.parse_args()
    
    main(args)

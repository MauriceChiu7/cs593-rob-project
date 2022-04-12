import argparse
import os
import torch
import pickle
import random
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
import numpy as np


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
                        nn.Linear(256, stateLength)
                     )

    def forward(self, x):
        x = self.fc(x)
        return x


# TODO: normalize inputs and outputs
def train():
    modelName = "A1"
    print("LOADING IN DATA FROM PICKLED FILE ...")
    folder = "saData_Mult/"
    mult = 6
    iterations = 150
    epochs = 5
    episodes = 50
    # with open(f'{folder}run_I{iterations}_E{epochs}_Eps{episodes}.pkl', 'rb') as f:
    #     allData = pickle.load(f)
    with open(f'{folder}MULT{mult}_run_I{iterations}_E{epochs}_Eps{episodes}.pkl', 'rb') as f:
        allData = pickle.load(f)
    print("FINISHED LOADING DATA!")

    print("TRAINING NEURAL NET ... ")
    stateLength = 15
    actionLength = 12
    # Shuffle state-action pairs
    random.shuffle(allData)
    # allData = allData[:1000]
    trainRatio = 0.7
    length = int(len(allData)*trainRatio)
    trainData = allData[:length]
    testData = allData[length:]

    # Initialize neural network
    neuralNet = NeuralNetwork(stateLength, actionLength)
    optimizer = torch.optim.Adam(neuralNet.parameters(), lr=0.01)

    mse = nn.MSELoss()
    epochs = 20
    trainMSE = []
    testMSE = []
    for epoch in range(epochs):
        print(f"THIS IS EPOCH: {epoch}")
        currTrainMSE = []
        random.shuffle(trainData)
        random.shuffle(testData)
        print("TRAINING...")
        for t in tqdm(trainData):
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

        currTestMSE = []
        print("TESTING...")
        for t in tqdm(testData):
            state_action = t[:stateLength+actionLength]
            nnTarget = t[stateLength+actionLength:]
            # Pass into Neural Net and get MSE Loss
            nnPred = neuralNet.forward(torch.Tensor(state_action))
            loss = mse(nnPred, torch.Tensor(nnTarget))
            currTestMSE.append(loss.item())
        avgTestMSE = np.mean(currTestMSE)
        testMSE.append(avgTestMSE)
        print("Testing Loss: ", avgTestMSE)

    # Save model
    print("FINISEHD TRAINING!")
    print("SAVING MODEL ... ")
    modelFolder = "./mult_models/"
    if not os.path.exists(modelFolder):
        # create directory if not exist
        os.makedirs(modelFolder)
    
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
    train()

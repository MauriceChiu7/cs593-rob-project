import argparse
import os
import torch
import pickle
import random
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
import numpy as np

# MAXES AND MINS FOR STATES AND ACTIONS. GOT HIS FROM maxStateAction.py AND HARDCODED IT TO RUN FASTER. 
minState = [-1.02153601, -1.1083865, 0.04107661, -1.04536732, -1.03269277 , 0.02834534, -1.2568753, -0.91284767, 0.06280416, -1.25270735, -0.83711144, 0.06090062, -1.11715088, -0.97780094, 0.06341114]
minAction = [-0.80285144, -1.04719758, -2.69653368, -0.80285144, -1.04719758, -2.69653368, -0.80285144, -1.04719758, -2.69653368, -0.80285144, -1.04719758, -2.69653368]
stateRange = [5.35852929, 2.55355967, 0.5749573, 5.37398911, 2.57010978, 0.59185147, 5.24135348, 2.27949571, 0.50033069, 5.2288108300000005, 2.29599363, 0.51464844, 5.2859171400000005, 2.43478941, 0.40024636]
actionRange = [1.60570288, 5.2359879, 1.78023583, 1.60570288, 5.2359879, 1.78023583, 1.60570288, 5.2359879, 1.78023583, 1.60570288, 5.2359879, 1.78023583]

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


def normalizeState(state):
    global minState, stateRange
    diff = np.subtract(state, minState)
    normalState = diff/stateRange
    return normalState.tolist()

def normalizeAction(action):
    global minAction, actionRange
    diff = np.subtract(action, minAction)
    normalAction = diff/actionRange
    return normalAction.tolist()

    
# TODO: normalize inputs and outputs
def train():
    # define model name
    modelName = "V1_Model"

    # create model folder
    modelFolder = f"./mult_models/{modelName}/"
    if not os.path.exists(modelFolder):
        # create directory if not exist
        os.makedirs(modelFolder)

    # Load in train and test data
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
    # split into train and test data
    trainRatio = 0.9
    length = int(len(allData)*trainRatio)
    trainData = allData[:length]
    testData = allData[length:]

    # Initialize neural network
    neuralNet = NeuralNetwork(stateLength, actionLength)
    optimizer = torch.optim.Adam(neuralNet.parameters(), lr=0.01)

    mse = nn.MSELoss()
    epochs = 50
    trainMSE = []
    testMSE = []
    for epoch in range(epochs):
        print(f"THIS IS EPOCH: {epoch}")
        currTrainMSE = []
        random.shuffle(trainData)
        random.shuffle(testData)
        print("TRAINING...")
        for t in tqdm(trainData):
            normS1 = normalizeState(t[:stateLength])
            normA = normalizeAction(t[stateLength:stateLength+actionLength])
            normS2 = normalizeState(t[stateLength+actionLength:])
            state_action = normS1 + normA
            nnTarget = normS2
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

        torch.save(neuralNet.fc.state_dict(), modelFolder + f"{modelName}_epoch{epoch}.pt")

        currTestMSE = []
        print("TESTING...")
        for t in tqdm(testData):
            normS1 = normalizeState(t[:stateLength])
            normA = normalizeAction(t[stateLength:stateLength+actionLength])
            normS2 = normalizeState(t[stateLength+actionLength:])
            state_action = normS1 + normA
            nnTarget = normS2
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

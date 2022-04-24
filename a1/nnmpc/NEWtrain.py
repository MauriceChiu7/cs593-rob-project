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
minState = [-1.0297287398795225, -1.6644083654026904, -0.015584937923837375, -1.0021420251033062, -1.6705273018242133, -0.008343590749213488, -0.8590960085231067, -1.7285345399750789, -0.02795621764922601, -1.0578811858705537, -1.641913003325433, -0.0025479132536383314, -1.068227497165033, -1.6219328574424108, -0.022971474016212648, -0.8770603655513525, -1.6987640819269179, -0.039889400106911624, -1.2568752990223655, -1.3210953331392936, 0.011956009963431019, -1.2557680387705799, -1.3214811303046041, 0.005454480898304741, -1.2829497394833986, -1.3814368399802714, -0.04579027727923243, -1.2527073515093285, -1.298596686420213, 0.017226176287621603, -1.2565594381598746, -1.284249218438306, -0.011482361365547256, -1.3007383470144487, -1.418266284133236, -0.03336962572266933, -1.1171508752374961, -1.4930602742729848, 0.03155428938255963]
minAction = [-0.80285144, -1.04719758, -2.69653368, -0.80285144, -1.04719758, -2.69653368, -0.80285144, -1.04719758, -2.69653368, -0.80285144, -1.04719758, -2.69653368]
stateRange = [4.761497558240923, 3.7112114299483405, 0.7519433260297952, 4.776578272504329, 3.712938386985593, 0.7578246885761932, 4.775229545743272, 3.9290905825926625, 0.8850212907837247, 4.731175528692843, 3.7108143173557067, 0.699352191476181, 4.710560939573208, 3.704278615778671, 0.7299159716380068, 4.638382378077592, 3.8709040566524915, 0.8448035247828886, 4.858898273711912, 3.050115272740485, 0.6355866880288158, 4.918093853280212, 3.0308168974654284, 0.6695222803466775, 5.033055561619363, 3.232053531269027, 0.7371904560779658, 4.762149383040643, 3.0497447957559833, 0.6084018678140535, 4.707957095207612, 3.046731769249619, 0.6718921218839282, 4.844659306880555, 3.3087431860563856, 0.8282433731473829, 4.72570788735529, 3.4035906171442045, 0.5369954942580496]
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
    modelFolder = f"./NEWmult_models/{modelName}/"
    if not os.path.exists(modelFolder):
        # create directory if not exist
        os.makedirs(modelFolder)

    # Load in train and test data
    print("LOADING IN DATA FROM PICKLED FILE ...")
    mult = 6
    iterations = 150
    epochs = 5
    episodes = 50
  
    allData = []
    folder = "./NEWmultActions_I150_E5_Eps50/MultRun_5.pkl"
    with open(folder, 'rb') as f:
        allData = pickle.load(f)


    print("FINISHED LOADING DATA!")

    print("TRAINING NEURAL NET ... ")
    stateLength = 39
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
            normS1 = normalizeState(t[0])
            normA = normalizeAction(t[1])
            normS2 = normalizeState(t[2])
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
            normS1 = normalizeState(t[0])
            normA = normalizeAction(t[1])
            normS2 = normalizeState(t[2])
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
    

    graphFolder = "./NEWgraphs/"
    if not os.path.exists(graphFolder):
        # create directory if not exist
        os.makedirs(graphFolder)
    plt.savefig(f"{graphFolder}{modelName}_MSE_Results.png")

    plt.show()
  

if __name__ == '__main__':
    train()

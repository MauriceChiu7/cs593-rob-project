import argparse
import os
import torch
import pickle
import random
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
import numpy as np


# print("LOADING IN DATA FROM PICKLED FILE ...")

# folder = "saData_Mult/"
# mult = 6
# iterations = 150
# epochs = 5
# episodes = 50

# with open(f'{folder}MULT{mult}_run_I{iterations}_E{epochs}_Eps{episodes}.pkl', 'rb') as f:
#     allData = pickle.load(f)

# print("FINISHED LOADING DATA!")

# # loop through data to get an array of all states and all actions
# stateLen = 15
# actionLen = 12
# states = []
# actions = []

# for dat in allData:
#     s1 = dat[:stateLen]
#     a = dat[stateLen:stateLen+actionLen]
#     s2 = dat[stateLen+actionLen:]
#     states.append(s1)
#     states.append(s2)
#     actions.append(a)

# maxState = np.amax(states, axis=0)
# minState = np.amin(states, axis = 0)
# maxAction = np.amax(actions, axis=0)
# minAction = np.amin(actions, axis = 0)

# print(maxState)
# print(minState)
# print(maxAction)
# print(minAction)

maxState = [4.33699328, 1.44517317, 0.61603391, 4.32862179, 1.53741701, 0.62019681, 3.98447818, 1.36664804, 0.56313485, 3.97610348, 1.45888219, 0.57554906, 4.16876626, 1.45698847, 0.4636575 ]
minState = [-1.02153601, -1.1083865, 0.04107661, -1.04536732, -1.03269277 , 0.02834534, -1.2568753, -0.91284767, 0.06280416, -1.25270735, -0.83711144, 0.06090062, -1.11715088, -0.97780094, 0.06341114]
maxAction = [0.80285144, 4.18879032, -0.91629785, 0.80285144, 4.18879032, -0.91629785, 0.80285144, 4.18879032, -0.91629785, 0.80285144, 4.18879032, -0.91629785]
minAction = [-0.80285144, -1.04719758, -2.69653368, -0.80285144, -1.04719758, -2.69653368, -0.80285144, -1.04719758, -2.69653368, -0.80285144, -1.04719758, -2.69653368]
stateRange = [5.35852929, 2.55355967, 0.5749573, 5.37398911, 2.57010978, 0.59185147, 5.24135348, 2.27949571, 0.50033069, 5.2288108300000005, 2.29599363, 0.51464844, 5.2859171400000005, 2.43478941, 0.40024636]
actionRange = [1.60570288, 5.2359879, 1.78023583, 1.60570288, 5.2359879, 1.78023583, 1.60570288, 5.2359879, 1.78023583, 1.60570288, 5.2359879, 1.78023583]

# stateRange = np.subtract(maxState, minState)
# print(stateRange.tolist())

# actionRange = np.subtract(maxAction, minAction)
# print(actionRange.tolist())
import pybullet as p
import torch
import time
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt

with open('results/100graph.pkl', 'rb') as f:
    error = pickle.load(f)

plt.plot(error)
plt.title("Error of Top Action Sequence Through Epochs")
plt.ylabel("Average Error")
plt.xlabel("Epoch Number")
plt.savefig("results/ErrorThroughEpochs100L")
plt.show()
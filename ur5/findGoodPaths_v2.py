# finds good paths for the ur5, prints out statistics

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

goodPaths = []

for filename in os.listdir("./trainingData"):
    path = filename.split("_")[1].split(".")[0]
    
    print(filename)
    
    
    with open(f"./trainingData/{filename}", 'rb') as f:
        traj = pickle.load(f)
    
    print(len(traj))
    if len(traj) < 3:
        goodPaths.append(int(path))

goodPaths.sort()
print(f"Good Paths: {goodPaths}")

percentage = len(goodPaths)/len(os.listdir("./trainingData"))
print(f"{len(goodPaths)} out of {len(os.listdir('./trainingData'))} are good paths. {percentage*100}%")


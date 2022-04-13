# from sklearn.metrics import mean_squared_error as mse
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

goodPaths = []

for filename in os.listdir("./trainingDataWithEE"):
    path = filename.split("_")[1].split(".")[0]
    
    # print(filename)
    # print(path)
    
    trajFile = f"./error/traj_{path}.pkl"
    simFile = f"./error/finalEePos_{path}.pkl"
    with open(trajFile, 'rb') as f:
        traj = pickle.load(f)
    with open(simFile, 'rb') as f:
        sim = pickle.load(f)
    
    # print(traj)
    # print(sim)
    # error = mse(traj, sim)
    error = np.square(np.subtract(traj,sim)).mean()
    # print(error)

    print(f"{filename} has MSE: {error}")

    if not os.path.exists(f"./figures/traj_comp_{path}.png"):
        print(f"creating figure for path {path}")
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_aspect('auto')
        ax.set_xlim([-0.8,0.8])
        ax.set_ylim([-0.8,0.8])
        ax.set_zlim([-0,0.8])
        ax.scatter3D(traj[:,0], traj[:,1], traj[:,2], c='green')
        ax.scatter3D(sim[:,0], sim[:,1], sim[:,2], c='red')
        plt.figtext(0.1, 0.95, f"path {path}")
        plt.savefig(f"./figures/traj_comp_{path}.png")
        plt.close()

    if error <= 0.02:
        goodPaths.append(int(path))

goodPaths.sort()
print(f"Good Paths: {goodPaths}")

percentage = len(goodPaths)/len(os.listdir("./trainingDataWithEE"))
print(f"{len(goodPaths)} out of {len(os.listdir('./trainingDataWithEE'))} are good paths. {percentage*100}%")


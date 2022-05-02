import pickle
import matplotlib.pyplot as plt
# import matplotlib
# import matplotlib.patches as mpatches
# import numpy as np
# from mpl_toolkits import mplot3d
# from mpl_toolkits.mplot3d import Axes3D 
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

colors = ["Orange", "Blue", "Green", "Black", "Red", "Brown", "Olive", "Cyan", "Gray", "Purple"]
grays = ["Gray", "Gray", "Gray", "Gray", "Gray", "Gray", "Gray", "Gray", "Gray", "Gray"]

def main():
    '''
    Description: plots and shows the trajectory from the pickle file.

    Input: None

    Returns: None
    '''
    path = 37
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_aspect('auto')
    ax.set_xlim([-0.8,0.8])
    ax.set_ylim([-0.8,0.8])
    ax.set_zlim([-0,0.8])
    trajFile = f"./error_old/traj_{path}.pkl"
    simFile = f"./error_old/finalEePos_{path}.pkl"
    with open(trajFile, 'rb') as f:
        traj = pickle.load(f)
    with open(simFile, 'rb') as f:
        sim = pickle.load(f)
    print(traj)
    print(sim)
    ax.scatter3D(traj[:,0], traj[:,1], traj[:,2], c='green')
    ax.scatter3D(sim[:,0], sim[:,1], sim[:,2], c='red')
    plt.figtext(0.1, 0.95, f"path {path}")
    plt.show()

main()
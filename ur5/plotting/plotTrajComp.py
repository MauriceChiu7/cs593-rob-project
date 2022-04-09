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
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_aspect('auto')
    ax.set_xlim([-0.8,0.8])
    ax.set_ylim([-0.8,0.8])
    ax.set_zlim([-0,0.8])
    trajFile = "./error/traj.pkl"
    simFile = "./error/finalEePos.pkl"
    with open(trajFile, 'rb') as f:
        traj = pickle.load(f)
    with open(simFile, 'rb') as f:
        sim = pickle.load(f)
    print(traj)
    print(sim)
    ax.scatter3D(traj[:,0], traj[:,1], traj[:,2], c='green')
    ax.scatter3D(sim[:,0], sim[:,1], sim[:,2], c='red')
    plt.show()

main()
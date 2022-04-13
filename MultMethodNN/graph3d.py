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
    path = 37
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_aspect('auto')
    
    actualFile = f"./trajectories/V1_run_I150_E3_Eps70.pkl_ACTUAL"
    predFile = f"./trajectories/V1_run_I150_E3_Eps70_NNPREDICTED.pkl"
    with open(actualFile, 'rb') as f:
        act = pickle.load(f)
    with open(predFile, 'rb') as f:
        pred = pickle.load(f)
    

    for i in act: 
        ax.scatter3D(i[0], i[1], i[2], c='green')
    for j in pred:
        ax.scatter3D(j[0], j[1], j[2], c='red')

    start = [0.012731, 0.002186, 0.42040155674978175]
    ax.scatter3D(start[0], start[1], start[2], c='blue')
    plt.figtext(0.1, 0.95, f"path {path}")
    plt.show()

main()
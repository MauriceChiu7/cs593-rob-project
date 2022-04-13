import pickle
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main(args):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_aspect('auto')
    
    actualFile = args.act
    predFile = args.pred
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
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Playback Actions for A1')
    parser.add_argument('--act', type=str, help="file to actual states")
    parser.add_argument('--pred', type=str, help="file to predicted states")
    args = parser.parse_args()
    
    main(args)

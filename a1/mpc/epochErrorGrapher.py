import pybullet as p
import torch
import time
import numpy as np
import math
import pickle
import sys
import matplotlib.pyplot as plt


def main():
    """
    Plots the error graphs generated from run.py. Make sure to uncomment 
    lines in run.py to generate the data.
    """

    graphPath = sys.argv[1]
    with open(graphPath, 'rb') as f:
        error = pickle.load(f)

    plt.plot(error)
    plt.title("Error of Top Action Sequence Through Epochs")
    plt.ylabel("Average Error")
    plt.xlabel("Epoch Number")
    plt.savefig("results/ErrorThroughEpochs100L")
    plt.show()

if __name__ == '__main__':
    main()
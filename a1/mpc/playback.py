# playback the saved data
import argparse
import pybullet as p
import torch
import time
import numpy as np
import math
import pickle


robotHeight = 0.420393


def loadDog():
    """
    Description: loads A1 and the environment

    Returns:
    :quadruped - {Int} ID of the robot
    :jointIds - {List of Int} list of joint IDs
    
    """

    p.connect(p.GUI)
    plane = p.loadURDF("../../unitree_pybullet/data/plane.urdf")
    p.setGravity(0,0,-9.8)
    p.setTimeStep(1./50)
    urdfFlags = p.URDF_USE_SELF_COLLISION
    quadruped = p.loadURDF("../../unitree_pybullet/data/a1/urdf/a1.urdf",[0,0,0.48],[0,0,0,1], flags = urdfFlags,useFixedBase=False)
    lower_legs = [2,5,8,11]
    for l0 in lower_legs:
        for l1 in lower_legs:
            if (l1>l0):
                enableCollision = 1
                p.setCollisionFilterPair(quadruped, quadruped, 2,5,enableCollision)
    jointIds=[]
    paramIds=[]

    for j in range (p.getNumJoints(quadruped)):
        p.changeDynamics(quadruped,j,linearDamping=0, angularDamping=0)
        info = p.getJointInfo(quadruped,j)
        jointName = info[1]
        jointType = info[2]
        if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
            jointIds.append(j)
    p.getCameraImage(480,320)
    p.setRealTimeSimulation(0)

    return quadruped, jointIds


def main(args):
    """
    Executes a set of actions in Pybullet environment. Specify file on the command line.
    """

    quadruped, jointIds = loadDog()
    for _ in range(100):
        p.stepSimulation()

    # PLAYBACK ACTIONS
    with open(args.file, 'rb') as f:
        actions = pickle.load(f)

    if args.mode == "mpc":
        for a in actions:
            p.setJointMotorControlArray(quadruped, jointIds, p.POSITION_CONTROL, a)
            p.stepSimulation()
            time.sleep(0.1)
    # Playback stuff produced from generateData
    elif args.mode == "gendata":
        stateLength = 15
        actionLength = 12

        for t in actions:
            action = t[stateLength:stateLength + actionLength]
            p.setJointMotorControlArray(quadruped, jointIds, p.POSITION_CONTROL, action)
            p.stepSimulation()
            time.sleep(0.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Playback Actions for A1')
    parser.add_argument('--mode', type=str, default="mpc", help="mpc or gendata")
    parser.add_argument('--file', type=str, default="results/BESTRun.pkl", help="path to file")
    args = parser.parse_args()
    
    main(args)

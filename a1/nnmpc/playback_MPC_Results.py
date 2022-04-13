import argparse
import pybullet as p
import torch
import time
import numpy as np
import math
import pickle
import os

robotHeight = 0.420393


def getState(quadruped):
    # ideal height for dog to maintain
    global robotHeight
    hips = []
    # goal point for dog to reach
    goalPoint = [10,0, robotHeight]    
    # [FR, FL, RR, RL]
    hipIds = [2,6,10,14]
    for id in hipIds:
        hips.append(p.getLinkState(quadruped, id)[0])
    pitchR = abs(hips[0][2] - hips[2][2])
    pitchL = abs(hips[1][2] - hips[3][2])
    rollF = abs(hips[0][2] - hips[1][2])
    rollR = abs(hips[2][2] - hips[3][2])
    yawR = abs(hips[0][1] - hips[2][1])
    yawL = abs(hips[1][1] - hips[3][1])
    pos = (p.getLinkState(quadruped, 0)[0])
    distance = math.dist(pos, goalPoint)**2
    heightErr = abs(robotHeight - pos[2])
    state = torch.Tensor([pitchR, pitchL, rollF, rollR, yawR, yawL, distance, heightErr])
    return state 

def getReward(action, jointIds, quadruped):
    # print(action)
    # p.setJointMotorControlArray(quadruped, jointIds, p.POSITION_CONTROL, action)
    # p.stepSimulation()
    state = getState(quadruped)
    w = torch.Tensor([2000,2000,300,300,300,300,2,3000])
    reward = (w*state).sum().numpy()
    if state[-1] > 0.25:
        reward += 1000
    return reward

def loadDog():
    p.connect(p.GUI)
    plane = p.loadURDF("plane.urdf")
    p.setGravity(0,0,-9.8)
    p.setTimeStep(1./50)
    urdfFlags = p.URDF_USE_SELF_COLLISION
    quadruped = p.loadURDF("a1/urdf/a1.urdf",[0,0,0.48],[0,0,0,1], flags = urdfFlags,useFixedBase=False)
    lower_legs = [2,5,8,11]
    for l0 in lower_legs:
        for l1 in lower_legs:
            if (l1>l0):
                enableCollision = 1
                p.setCollisionFilterPair(quadruped, quadruped, 2,5,enableCollision)
    jointIds=[]
    paramIds=[]
    maxForceId = p.addUserDebugParameter("maxForce",0,100,20)
    for j in range (p.getNumJoints(quadruped)):
        p.changeDynamics(quadruped,j,linearDamping=0, angularDamping=0)
        info = p.getJointInfo(quadruped,j)
        jointName = info[1]
        jointType = info[2]
        if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
            jointIds.append(j)
    p.getCameraImage(480,320)
    p.setRealTimeSimulation(0)
    joints=[]
    return maxForceId, quadruped, jointIds

def main(args):
    maxForceId, quadruped, jointIds = loadDog()
    for _ in range(100):
        p.stepSimulation()

    # THIS IS FOR PLAYBACK FROM THE TESTNNMPC FOLDER
    folder = 'NN_MPC_Action_Results/'
    fileName = args.file

    with open(folder+fileName, 'rb') as f:
        actions = pickle.load(f)

    centerTrajectory = []
    length = int(len(actions)/12)
    for x in range(length):
        m = x+1
        p.setJointMotorControlArray(quadruped, jointIds, p.POSITION_CONTROL, actions[(m-1)*12:m*12])
        p.stepSimulation()
        time.sleep(0.15)
        centerTrajectory.append(p.getLinkState(quadruped, 0)[0])

    trajPath = 'trajectories/'
    if not os.path.exists(trajPath):
        # create directory if not exist
        os.makedirs(trajPath)

    trajFile = trajPath + fileName + "_ACTUAL"

    with open(trajFile, 'wb') as f:
        pickle.dump(centerTrajectory, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Playback Actions From run.py. Also Generates actual trajectory File')
    parser.add_argument('--file', type=str, default='V1_run_I150_E3_Eps70.pkl', help="file name from folder NN_MPC_Action_Results")
    args = parser.parse_args()
    
    main(args)

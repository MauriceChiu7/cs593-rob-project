"""
THIS FILE RUNS WHAT PATHS THE MPC ALGORITHM THAT USED PYBULLET CONSIDERED DURING A PARTICULAR RUN. 
IN OTHER WORDS, IT PLAYSBACK TO US THE TOPK PATH CONSIDERED BY THE MPC ALGORITHM AS A CERTAIN POINT
"""
import argparse
import pybullet as p
import torch
import time
import numpy as np
import math
import pickle

def loadDog():
    # class Dog:
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
                # print("collision for pair",l0,l1, p.getJointInfo(quadruped,l0)[12],p.getJointInfo(quadruped,l1)[12], "enabled=",enableCollision)
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

    # ___________________________________________________________
    # ___________________________________________________________
    # CHANGE PARAMETERS HERE FOR PLAYBACK
    multRun = 0
    Iter = 149
    Top = 1
    overallIterations = 150
    overallEpochs = 5
    overallEpisodes = 50
    # ___________________________________________________________
    # ___________________________________________________________

    folder = f'multActions_I{args.overallIters}_E{args.overallEpochs}_Eps{args.overallEps}/'

    # LOAD IN ACTUAL RUN THAT THIS MULTRUN PRODUCED
    with open(f'{folder}run_I{args.overallIters}_E{args.overallEpochs}_Eps{args.overallEps}_Mult{args.runNum}.pkl','rb') as f:
        startActions = pickle.load(f)

    # LOAD IN A TOPK PATH CONSIDERED BY THE MPC ALGORITHM
    with open(f'{folder}/MultRun_{args.runNum}_Iter_{Iter}_Top{Top}.pkl', 'rb') as f:
        actions = pickle.load(f)

    # RUN THE FIRST ITER ACTIONS ON THE MULTRUN FILE (AS THIS WILL SET THE ROBOT UP AT THE CORRECT SPOT)
    for x in range(Iter):
        p.setJointMotorControlArray(quadruped, jointIds, p.POSITION_CONTROL, startActions[x])
        p.stepSimulation()
        time.sleep(0.05)

    # RUN WHAT THE MPC ALGORITHM CONSIDERED AS A TOPK PATH FROM THIS POINT ON
    for x in range(int(len(actions)/12)):
        m = x+1
        p.setJointMotorControlArray(quadruped, jointIds, p.POSITION_CONTROL, actions[(m-1)*12:m*12])
        p.stepSimulation()
        # print(getState(quadruped))
        # print(getReward(action, jointIds, quadruped))
        time.sleep(0.05)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Playback Training Data Generated by train.py to View What "Ground Truth" Looks Like.')
    parser.add_argument('--overallIters', type=int, default=150, help="Overall Iterations of the Run you are looking at")
    parser.add_argument('--overallEpochs', type=int, default=5, help="Overall Epochs of the Run you are looking at")
    parser.add_argument('--overallEps', type=int, default=50, help="Overall Episodes of the Run you are looking at")
    parser.add_argument('--runNum', type=int, default=0, help="Which Run you want to play back")
    parser.add_argument('--iter', type=int, default=40, help="Iteration from which top episodes will be chosen to play back")
    parser.add_argument('--top', type=int, default=1, help="The top x episode you want to playback")
    args = parser.parse_args()
    
    main(args)
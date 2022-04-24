"""
Since the state information generated before wasn't specific enough to train the NN, this file reruns our saved playback files and captures states in more detail.
More specifically, originally, we were only saving the cartesian coordinates of the hips and center of the A1 robot dog. 
This file will rerun the saved playback actions for each run and instead, save the cartesian coordinates of every joint for the A1 robot dog.
"""
import argparse
import pybullet as p
import torch
import time
import numpy as np
import math
import pickle

def getState(quadruped):
    state = []
    # [FR, FL, RR, RL]
    jointIds = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16]
    for id in jointIds:
        state.extend(p.getLinkState(quadruped, id)[0])    
    # Get body
    state.extend(p.getLinkState(quadruped, 0)[0])
    return state

def setUp():
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

    for _ in range(100):
        p.stepSimulation()

    return maxForceId, quadruped, jointIds

def main():
    Mult = 6
    Iters = 150
    Top = 2
    folder = 'multActions_I150_E5_Eps50/'
    finalSAPairs = []

    for mult in range(Mult):
        print(f"MULT {mult}")
        for i in range(Iters):
            print(f"ITERATION {i}")
            for t in range(Top):
                maxForceId, quadruped, jointIds = setUp()
                print(jointIds)
                # LOAD IN ACTUAL RUN THAT THIS MULTRUN PRODUCED
                print(f"THIS IS MULT : {mult}!!!!!!!!!")
                with open(f'{folder}run_I150_E5_Eps50_Mult{mult}.pkl','rb') as f:
                    startActions = pickle.load(f)
                # LOAD IN A TOPK PATH CONSIDERED BY THE MPC ALGORITHM
                with open(f'{folder}/MultRun_{mult}_Iter_{i}_Top{t+1}.pkl', 'rb') as f:
                    actions = pickle.load(f)

                # RUN THE FIRST ITER ACTIONS ON THE MULTRUN FILE (AS THIS WILL SET THE ROBOT UP AT THE CORRECT SPOT)
                for x in range(i):
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
                p.disconnect()

if __name__ == '__main__':
    main()
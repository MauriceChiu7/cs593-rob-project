from ast import Pass
import pybullet as p
import torch
import time
import numpy as np
import math
import pickle

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
    # class Dog:
    p.connect(p.GUI)
    plane = p.loadURDF("plane.urdf")
    p.setGravity(0,0,-9.8)
    p.setTimeStep(1./50)
    #p.setDefaultContactERP(0)
    #urdfFlags = p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
    urdfFlags = p.URDF_USE_SELF_COLLISION
    quadruped = p.loadURDF("a1/urdf/a1.urdf",[0,0,0.48],[0,0,0,1], flags = urdfFlags,useFixedBase=False)

    #enable collision between lower legs
    # for j in range (p.getNumJoints(quadruped)):
            # print(p.getJointInfo(quadruped,j))

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
        # print(info)
        jointName = info[1]
        jointType = info[2]
        if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
            jointIds.append(j)

    # print(jointIds)

    p.getCameraImage(480,320)
    p.setRealTimeSimulation(0)

    joints=[]
    return maxForceId, quadruped, jointIds


maxForceId, quadruped, jointIds = loadDog()
for _ in range(100):
    p.stepSimulation()

# with open('results/run_I300_E5_Eps150.pkl', 'rb') as f:
#     actions = pickle.load(f)

# realActions = actions[:-150]

# with open(f"results/GOODRUN.pkl", 'wb') as f:
#     pickle.dump(realActions, f)

# exit()

# print(actions)

with open('results/run_I200_E4_Eps30.pkl', 'rb') as f:
    actions = pickle.load(f)

# print(len(actions))
# exit()

for action in actions:
    p.setJointMotorControlArray(quadruped, jointIds, p.POSITION_CONTROL, action)
    p.stepSimulation()
    # print(getState(quadruped))
    print(getReward(action, jointIds, quadruped))
    time.sleep(0.05)

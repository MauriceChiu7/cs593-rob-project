from mimetypes import init
import pybullet as p
import time
import os
import sys
import pybullet_data
import numpy as np
import torch

# class Dog:
def loadA1():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), 0, 0, 0)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1./500)

    urdfFlags = p.URDF_USE_SELF_COLLISION
    quadruped = p.loadURDF("a1/urdf/a1.urdf",[0,0,0.48],[0,0,0,1], flags = urdfFlags,useFixedBase=False)

    #enable collision between lower legs
    lower_legs = [2,5,8,11]
    for l0 in lower_legs:
        for l1 in lower_legs:
            if (l1>l0):
                enableCollision = 1
                # print("collision for pair",l0,l1, p.getJointInfo(quadruped,l0)[12],p.getJointInfo(quadruped,l1)[12], "enabled=",enableCollision)
                p.setCollisionFilterPair(quadruped, quadruped, 2, 5, enableCollision)

    jointIds=[]

    maxForceId = p.addUserDebugParameter("maxForce",0,100,20)

    # Set resistance to none
    for j in range(p.getNumJoints(quadruped)):
        p.changeDynamics(quadruped, j, linearDamping=0, angularDamping=0)
        info = p.getJointInfo(quadruped,j)
        jointName = info[1]
        jointType = info[2]
        if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
            jointIds.append(j)

    return urdfFlags, quadruped


def getEnvInfo(quadruped):
    initPosition, initOrientation = p.getBasePositionAndOrientation(quadruped)
    envInfo = {
        'position': initPosition, 
        'orientation': initOrientation
    }

    return envInfo


def getRobotInfo(quadruped):
    numJoints = p.getNumJoints(quadruped)
    jointInfo = {}
    jointStates = {}
    for j in range(numJoints):
        jointInfo[j] = p.getJointInfo(quadruped, j)
        jointStates[j] = p.getJointState(quadruped, j)    # jointPosition, jointVelocity, jointReactionForces, appliedJointMotorTorque

    return jointInfo, jointStates


def simulate():
    # setup for simulation
    p.getCameraImage(480,320)
    p.setRealTimeSimulation(0)

    while(1):
        p.stepSimulation()


def main():
    urdfFlags, quadruped = loadA1()
    envInfo = getEnvInfo(quadruped)
    jointInfo, jointStates = getRobotInfo(quadruped)

    #### Milestone 1 ####
    # print("\nEnvironment State Info ---------------------------------------------------------")
    # print(envInfo)

    # print("\nRobot Info ---------------------------------------------------------------")
    # print("\nJoint Info ----------------------------------------------------")
    # for id in jointInfo:
    #     print("Joint", id, ":", jointInfo[id])

    # print("\nJoint State ----------------------------------------------------")
    # for id in jointStates:
    #     print("Joint", id, ":", jointStates[id])
    ########

    ####### Milestone 2 ######

    # Outer loop
    # For each env_step
    # steps = 10
    # for env_step in steps:

    # Get G samples of normal distribution
    G = 1000
    plans = []
    mu = torch.tensor([0.]*12)
    sigma = torch.tensor([1.]*11 + [20.])

    for _ in range(G):
        sam = torch.normal(mean=mu, std=sigma)
        plans.append(sam)

    # Just using floating base state for now
    FR_hip_joint = p.getLinkState(quadruped, 2)[0]
    FL_hip_joint = p.getLinkState(quadruped, 6)[0]
    RR_hip_joint = p.getLinkState(quadruped, 10)[0]
    RL_hip_joint = p.getLinkState(quadruped, 14)[0]
    print('These are the FR, FL, RR, RL hip joints respectively: ', FR_hip_joint, FL_hip_joint, RR_hip_joint, RL_hip_joint)

    s0 = [FR_hip_joint, FL_hip_joint, RR_hip_joint, RL_hip_joint]

    pred_horiz = 10
    iters = 100
    states = []
    # for i in range(iters):
    #     for h in H:
            
    #         configuration = Configuration.from_revolute_values([-2.238, -1.153, -2.174, 0.185, 0.667, 0.])

    #         frame_WCF = robot.forward_kinematics(configuration)




    print("--------")
    exit()
    # while(1):
    #     with open("mocap.txt","r") as filestream:
    #         for line in filestream:
    #             maxForce = p.readUserDebugParameter(maxForceId)
    #             currentline = line.split(",")
    #             frame = currentline[0]
    #             t = currentline[1]
    #             joints=currentline[2:14]
    #             for j in range (12):
    #                 targetPos = float(joints[j])
    #                 p.setJointMotorControl2(quadruped, jointIds[j], p.POSITION_CONTROL, targetPos, force=maxForce)

    #             p.stepSimulation()
    #             for lower_leg in lower_legs:
    #                 #print("points for ", quadruped, " link: ", lower_leg)
    #                 pts = p.getContactPoints(quadruped,-1, lower_leg)
    #                 #print("num points=",len(pts))
    #                 #for pt in pts:
    #                 #    print(pt[9])
    #             time.sleep(1./500.)





    # Joint 0 : (0, b'floating_base', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'trunk', (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), -1)
    # Joint 1 : (1, b'imu_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'imu_link', (0.0, 0.0, 0.0), (-0.012731, -0.002186, -0.000515), (0.0, 0.0, 0.0, 1.0), 0)
    # Joint 2 : (2, b'FR_hip_joint', 0, 7, 6, 1, 0.0, 0.0, -0.802851455917, 0.802851455917, 20.0, 52.4, b'FR_hip', (1.0, 0.0, 0.0), (0.170269, -0.049186, -0.000515), (0.0, 0.0, 0.0, 1.0), 0)
    # Joint 3 : (3, b'FR_thigh_joint', 0, 8, 7, 1, 0.0, 0.0, -1.0471975512, 4.18879020479, 55.0, 28.6, b'FR_thigh', (0.0, 1.0, 0.0), (0.003311, -0.084415, -3.1e-05), (0.0, 0.0, 0.0, 1.0), 2)
    # Joint 4 : (4, b'FR_calf_joint', 0, 9, 8, 1, 0.0, 0.0, -2.69653369433, -0.916297857297, 55.0, 28.6, b'FR_calf', (0.0, 1.0, 0.0), (0.003237, -0.022327, -0.17267400000000002), (0.0, 0.0, 0.0, 1.0), 3)
    # Joint 5 : (5, b'FR_foot_fixed', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'FR_foot', (0.0, 0.0, 0.0), (-0.006435, 0.0, -0.09261200000000001), (0.0, 0.0, 0.0, 1.0), 4)
    # Joint 6 : (6, b'FL_hip_joint', 0, 10, 9, 1, 0.0, 0.0, -0.802851455917, 0.802851455917, 20.0, 52.4, b'FL_hip', (1.0, 0.0, 0.0), (0.170269, 0.044814, -0.000515), (0.0, 0.0, 0.0, 1.0), 0)
    # Joint 7 : (7, b'FL_thigh_joint', 0, 11, 10, 1, 0.0, 0.0, -1.0471975512, 4.18879020479, 55.0, 28.6, b'FL_thigh', (0.0, 1.0, 0.0), (0.003311, 0.084415, -3.1e-05), (0.0, 0.0, 0.0, 1.0), 6)
    # Joint 8 : (8, b'FL_calf_joint', 0, 12, 11, 1, 0.0, 0.0, -2.69653369433, -0.916297857297, 55.0, 28.6, b'FL_calf', (0.0, 1.0, 0.0), (0.003237, 0.022327, -0.17267400000000002), (0.0, 0.0, 0.0, 1.0), 7)
    # Joint 9 : (9, b'FL_foot_fixed', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'FL_foot', (0.0, 0.0, 0.0), (-0.006435, 0.0, -0.09261200000000001), (0.0, 0.0, 0.0, 1.0), 8)
    # Joint 10 : (10, b'RR_hip_joint', 0, 13, 12, 1, 0.0, 0.0, -0.802851455917, 0.802851455917, 20.0, 52.4, b'RR_hip', (1.0, 0.0, 0.0), (-0.195731, -0.049186, -0.000515), (0.0, 0.0, 0.0, 1.0), 0)
    # Joint 11 : (11, b'RR_thigh_joint', 0, 14, 13, 1, 0.0, 0.0, -1.0471975512, 4.18879020479, 55.0, 28.6, b'RR_thigh', (0.0, 1.0, 0.0), (-0.003311, -0.084415, -3.1e-05), (0.0, 0.0, 0.0, 1.0), 10)
    # Joint 12 : (12, b'RR_calf_joint', 0, 15, 14, 1, 0.0, 0.0, -2.69653369433, -0.916297857297, 55.0, 28.6, b'RR_calf', (0.0, 1.0, 0.0), (0.003237, -0.022327, -0.17267400000000002), (0.0, 0.0, 0.0, 1.0), 11)
    # Joint 13 : (13, b'RR_foot_fixed', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'RR_foot', (0.0, 0.0, 0.0), (-0.006435, 0.0, -0.09261200000000001), (0.0, 0.0, 0.0, 1.0), 12)
    # Joint 14 : (14, b'RL_hip_joint', 0, 16, 15, 1, 0.0, 0.0, -0.802851455917, 0.802851455917, 20.0, 52.4, b'RL_hip', (1.0, 0.0, 0.0), (-0.195731, 0.044814, -0.000515), (0.0, 0.0, 0.0, 1.0), 0)
    # Joint 15 : (15, b'RL_thigh_joint', 0, 17, 16, 1, 0.0, 0.0, -1.0471975512, 4.18879020479, 55.0, 28.6, b'RL_thigh', (0.0, 1.0, 0.0), (-0.003311, 0.084415, -3.1e-05), (0.0, 0.0, 0.0, 1.0), 14)
    # Joint 16 : (16, b'RL_calf_joint', 0, 18, 17, 1, 0.0, 0.0, -2.69653369433, -0.916297857297, 55.0, 28.6, b'RL_calf', (0.0, 1.0, 0.0), (0.003237, 0.022327, -0.17267400000000002), (0.0, 0.0, 0.0, 1.0), 15)
    # Joint 17 : (17, b'RL_foot_fixed', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'RL_foot', (0.0, 0.0, 0.0), (-0.006435, 0.0, -0.09261200000000001), (0.0, 0.0, 0.0, 1.0), 16)

    # simulate()


if __name__ == '__main__':
    main()


[1,2,5.4,123,41]
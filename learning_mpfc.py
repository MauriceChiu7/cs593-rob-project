import argparse
import csv
from datetime import datetime
import os
from pprint import pprint
import pybullet as p
import pybullet_data
import math
import numpy as np
import torch
import time

SIM_STEPS = 3
CTL_FREQ = 20    # Hz

"""
Loads pybullet environment with a horizontal plane and earth like gravity.
"""
def loadEnv():
    if args.verbose: print(f"\nloading environment...\n")
    if args.fast: 
        p.connect(p.DIRECT) 
    else: 
        p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    if args.robot == 'ur5':
        p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0.1])
    else:
        p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0])
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1./ CTL_FREQ / SIM_STEPS)
    # p.setRealTimeSimulation(1)
    if args.verbose: print(f"\n...environment loaded\n")

"""
Loads the UR5 robot.
"""
def loadUR5(activeJoints):
    p.resetDebugVisualizerCamera( cameraDistance=1.5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=(0,0,0))
    if args.verbose: print(f"\nloading UR5...\n")
    path = f"{os.getcwd()}/ur5pybullet"
    os.chdir(path) # Needed to change directory to load the UR5.
    uid = p.loadURDF(os.path.join(os.getcwd(), "./urdf/real_arm.urdf"), [0.0,0.0,0.0], p.getQuaternionFromEuler([0,0,0]), flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION)
    path = f"{os.getcwd()}/.."
    os.chdir(path) # Back to parent directory.
    # Enable collision for all link pairs.
    for l0 in range(p.getNumJoints(uid)):
        for l1 in range(p.getNumJoints(uid)):
            if (not l1>l0):
                enableCollision = 1
                # print("collision for pair",l0,l1, p.getJointInfo(uid,l0)[12],p.getJointInfo(uid,l1)[12], "enabled=",enableCollision)
                p.setCollisionFilterPair(uid, uid, l1, l0, enableCollision)
    jointsForceRange = getJointsForceRange(uid, activeJoints)
    if args.verbose: print(f"\n...UR5 loaded\n")
    return uid, jointsForceRange

"""
Loads the A1 robot.
"""
def loadA1():
    if args.verbose: print(f"\nloading A1...\n")
    path = f"{os.getcwd()}/unitree_pybullet"
    os.chdir(path) # Needed to change directory to load the A1.
    quadruped = p.loadURDF("./data/a1/urdf/a1.urdf",[0,0,0.48], p.getQuaternionFromEuler([0,0,0]), flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION)
    path = f"{os.getcwd()}/.."
    os.chdir(path) # Back to parent directory.
    # Enable collision between lower legs.
    lower_legs = [2,5,8,11]
    for l0 in lower_legs:
        for l1 in lower_legs:
            if (l1>l0):
                enableCollision = 1
                # print("collision for pair",l0,l1, p.getJointInfo(quadruped,l0)[12],p.getJointInfo(quadruped,l1)[12], "enabled=",enableCollision)
                p.setCollisionFilterPair(quadruped, quadruped, 2, 5, enableCollision)
    # maxForceId = p.addUserDebugParameter("maxForce",0,500,20)
    jointIds=[]
    # Set resistance to none
    for j in range(p.getNumJoints(quadruped)):
        p.changeDynamics(quadruped, j, linearDamping=0, angularDamping=0)
        info = p.getJointInfo(quadruped,j)
        jointName = info[1]
        jointType = info[2]
        if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
            jointIds.append(j)
    if args.verbose: print(f"activeJoints: {jointIds}")
    # Adding extra lateral friction to the feet.
    foot_fixed = [5, 9, 13, 17] # 5: FntRgt, 9: FntLft, 13: RarRgt, 17: RarLft
    for foot in foot_fixed:
        p.changeDynamics(quadruped, foot, lateralFriction=1)
    jointsForceRange = getJointsForceRange(quadruped, jointIds)
    if args.verbose: print(f"\n...A1 loaded\n")
    return quadruped, jointsForceRange, jointIds

"""
Gets the maxForce of each joint active joints.
"""
def getJointsMaxForce(uid, jointIds):
    jointsMaxForces = []
    for j in jointIds:
        jointInfo = p.getJointInfo(uid, j)
        jointsMaxForces.append(jointInfo[10])
    if args.verbose: print(f"jointsMaxForces: {jointsMaxForces}")
    if args.verbose: print(f"len(jointsMaxForces): {len(jointsMaxForces)}")
    return jointsMaxForces

"""
Gets the min and max force of all active joints.
"""
def getJointsForceRange(uid, jointIds):
    forces = getJointsMaxForce(uid, jointIds)
    jointsForceRange = []
    for f in forces:
        mins = -f
        maxs = f
        jointsForceRange.append((mins, maxs))
    if args.verbose: print(f"jointsForceRange: {jointsForceRange}")
    if args.verbose: print(f"len(jointsForceRange): {len(jointsForceRange)}")
    return jointsForceRange

def setUpJointsForceSlider(uid, jointIds):
    jointsForceRange = [(-300, 300)] * len(jointIds)
    # jointsForceRange = getJointsForceRange(uid, jointIds)
    jointsForceIds = []
    for i in range(len(jointsForceRange)):
        jointNum = f"joint {jointIds[i]}"
        forceIds = p.addUserDebugParameter(jointNum, jointsForceRange[i][0], jointsForceRange[i][1], 0)
        jointsForceIds.append(forceIds)
    return jointsForceIds

def applyAction(uid, jointIds, action):
    action = torch.tensor(action)
    torqueScalar = 1
    if args.robot == 'ur5':
        # correction = 105
        # action = torch.where(action > 0, torch.add(action, correction), torch.add(action, -correction))
        # torqueScalar = 15
        torqueScalar = 1
    else:
        # torqueScalar = 15
        torqueScalar = 1
    action = torch.mul(action, torqueScalar)
    if args.verbose: print(f"action applied: \n{action}")
    p.setJointMotorControlArray(uid, jointIds, p.TORQUE_CONTROL, forces=action)
    for _ in range(SIM_STEPS):
        p.stepSimulation()


def main():
    loadEnv()
    if args.robot == 'ur5':
        ACTIVE_JOINTS = [1,2,3,4,5,6,8,9]
        uid, jointsForceRange = loadUR5(ACTIVE_JOINTS)
        jointIds = ACTIVE_JOINTS
    else:
        uid, jointsForceRange, jointIds = loadA1()
    jointsForceIds = setUpJointsForceSlider(uid, jointIds)

    while 1:
        action = []
        for id in jointsForceIds:
            force = p.readUserDebugParameter(id)
            action.append(force)
        applyAction(uid, jointIds, action)

def playback():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CS 593-ROB - Project Milestone 2')
    parser.add_argument('-p', '--play', action='store_true', help='Set true to playback the recorded best actions.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Logs debug information.')
    parser.add_argument('-f', '--fast', action='store_true', help='Trains faster without GUI.')
    parser.add_argument('-r', '--robot', default='ur5', help='Choose which robot, "ur5" or "a1", to simulate or playback actions.')
    parser.add_argument('-d', '--debug', action='store_true', help='Displays debug information.')
    # parser.add_argument('-G', '--nplan', help='Number of plans to generate.')
    # parser.add_argument('-T', '--train', help='Number of iterations to train.')
    # parser.add_argument('-H', '--horizon', default=5, help='Set the horizon length.')
    args = parser.parse_args()
    
    if args.play:
        playback()
    else:
        # test()
        main()
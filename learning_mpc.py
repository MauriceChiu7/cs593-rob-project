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
ACTIVE_JOINTS = [1,2,3,4,5,6,8,9]
END_EFFECTOR_INDEX = 7 # The end effector link index.
ELBOW_INDEX = 3 # The end effector link index.

"""
Calculates the difference between two vectors.
"""
def diff(v1, v2):
    return [x1 - x2 for x1, x2 in zip(v1, v2)]

"""
Calculates the magnitude of a vector.
"""
def magnitude(v):
    return math.sqrt(sum([x*x for x in v]))

"""
Calculates distance between two vectors.
"""
def dist(p1, p2):
    return magnitude(diff(p1, p2))

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
    # p.setTimeStep(1./ CTL_FREQ / SIM_STEPS) # 1/60
    # p.setRealTimeSimulation()
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
    # quadruped = p.loadURDF("./data/a1/urdf/a1.urdf", [0,0,0.48], p.getQuaternionFromEuler([0,0,np.pi*2]), flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION)
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

def setupJointsForceSlider(uid, jointIds):
    # jointsForceRange = [(-500, 500)] * len(jointIds)
    jointsForceRange = [(-300, 300)] * len(jointIds)
    # jointsForceRange = getJointsForceRange(uid, jointIds)
    jointsForceIds = []
    for i in range(len(jointsForceRange)):
        jointNum = f"joint {jointIds[i]}"
        forceIds = p.addUserDebugParameter(jointNum, jointsForceRange[i][0], jointsForceRange[i][1], 0)
        jointsForceIds.append(forceIds)
    return jointsForceIds

def setupTimeStepSlider():
    timeStepId = p.addUserDebugParameter("TimeStep", 1./1000, 1./60, 1./500)
    simStepId = p.addUserDebugParameter("SimStep", 0, 50, 1)
    
    return timeStepId, simStepId

def costFunc(currMu, targ):
    cost = diff(currMu, targ)
    if args.verbose: print(f"...cost: {cost}")

    direction = 0
    if torch.lt(torch.tensor(cost), torch.zeros(1)): 
        direction = -1
    else:
        direction = 1
    magnitude = dist(currMu, targ)
    
    print(f"direction: {direction}, magnitude: {magnitude}")

    return direction, magnitude

def genActionSeqSetFromNormalDist(mu, sigma, G, H, jointsForceRange):
    # mins, maxes = np.array(jointsForceRange).T
    mins, maxes = jointsForceRange
    actionSeqSet = []
    for _ in range(G):
        samp = torch.normal(mu, sigma)
        actionSeqSet.append(torch.tensor(samp.reshape(-1)))
    actionSeqSet = torch.stack(actionSeqSet)
    if args.verbose: print(f"actionSeqSet:\n{actionSeqSet}")
    if args.verbose: print(f"actionSeqSet.size():\n{actionSeqSet.size()}")
    return actionSeqSet

def getState(uid):
    eePos = p.getLinkState(uid, END_EFFECTOR_INDEX)[0]
    elbowPos = p.getLinkState(uid, ELBOW_INDEX)[0]
    state = torch.Tensor([eePos, elbowPos])
    return state


def getConfig(uid, jointIds):
    config = []
    for id in jointIds:
        # print(p.getJointState(uid, id)[0])
        config.append(p.getJointState(uid, id)[0])
    EEPos = getState(uid)[0].tolist()
    config.append(EEPos[0])
    config.append(EEPos[1])
    config.append(EEPos[2])
    return config

def applyAction(uid, action):
    p.setJointMotorControlArray(uid, ACTIVE_JOINTS, p.POSITION_CONTROL, action)
    maxSimSteps = 150
    for s in range(maxSimSteps):
        p.stepSimulation()
        currConfig = getConfig(uid, ACTIVE_JOINTS)[0:8]
        action = torch.Tensor(action)
        currConfig = torch.Tensor(currConfig)
        error = torch.sub(action, currConfig)
        done = True
        for e in error:
            if e > 0.02:
                done = False
        if done:
            # print(f"reached position: \n{action}, \nwith target:\n{currConfig}, \nand error: \n{error} \nin step {s}")
            break

def randomGoal():
    # Generate random goal state for UR5
    x = np.random.uniform(-0.7, 0.7)
    y = np.random.uniform(-0.7, 0.7)
    z = np.random.uniform(0.1, 0.7)
    goalCoords = torch.Tensor([x, y, z])
    p.addUserDebugLine([0,0,0.1], goalCoords, [0,0,1])
    return goalCoords

def getJointsRange(uid, jointIds):
    jointsRange = []
    for a in jointIds:
        jointInfo = p.getJointInfo(uid, a)
        jointsRange.append((jointInfo[8], jointInfo[9]))
    return jointsRange

def main():
    timeStepId = loadEnv()
    uid, jointsForceRange = loadUR5(ACTIVE_JOINTS)
    jointIds = ACTIVE_JOINTS
    jointsForceIds = setupJointsForceSlider(uid, jointIds)
    timeStepId, simStepId = setupTimeStepSlider()
    jointsRange = getJointsRange(uid, ACTIVE_JOINTS)

    for i in range(100):
        random_positions = []
        for r in jointsRange:
            rand = np.random.uniform(r[0], r[1])
            random_positions.append(rand)
        applyAction(uid, random_positions)

        # time.sleep(1)

    exit()

    while 1:
        action = []
        debugTS = p.readUserDebugParameter(timeStepId)
        p.setTimeStep(debugTS) # 1/60
        debugSS = p.readUserDebugParameter(simStepId)
        for id in jointsForceIds:
            force = p.readUserDebugParameter(id)
            action.append(force)
        applyAction(uid, jointIds, action, int(debugSS))
    
    # N = 1
    # T = 120
    # G = 220
    # K = int(0.4 * G)
    # H = 5

    # # mu = torch.zeros(H, len(jointsForceRange)).flatten()
    # # sigma = (np.pi * 1e05) * torch.eye(len(mu))
    # mu = torch.zeros(1)
    # sigma = (np.pi * 100) * torch.eye(len(mu))
    # print(f"mu: \n{mu}")
    # print(f"sigma: \n{sigma}")

    # targ = [-250]

    # for n in range(N):
    #     actionSeqSet = genActionSeqSetFromNormalDist(mu, sigma, G, H, (-800, 800))
    #     for t in range(T):
    #         planCosts = []
    #         cost = 0
    #         for actionSeq in actionSeqSet:
    #             # ___LINE 5___
    #             # Calculate the cost of the state sequence.
    #             if args.verbose: print(f"\ncurrMu: {actionSeq}, targMu: {targ}")
    #             direction, magnitude = costFunc(actionSeq, targ)
                
    #             planCosts.append((actionSeq, direction, magnitude))

    #         # ___LINE 6___
    #         # Sort action sequences by cost.
    #         sortedActionSeqSet = sorted(planCosts, key = lambda x: x[2])

    #         # ___LINE 7___
    #         # Update normal distribution to fit top K action sequences.
    #         eliteActionSeqSet = []
    #         for eliteActionSeq in range(K):
    #             eliteActionSeqSet.append(sortedActionSeqSet[eliteActionSeq][0])
    #         eliteActionSeqSet = torch.stack(eliteActionSeqSet)

    #         mu = torch.mean(eliteActionSeqSet, dim=0)
    #         mu.add(torch.mul(direction, magnitude))
    #         sigma = torch.cov(eliteActionSeqSet.T)
    #         # sigma += .02 * torch.eye(len(mu)) # add a small amount of noise to the diagonal to replan to next target
    #         if args.verbose: print(f"mu for envStep {n}:\n{mu}")
    #         if args.verbose: print(f"sigma for envStep {n}:\n{sigma}")

    #         # ___LINE 8___
    #         # Replace bottom G-K sequences with better action sequences.
    #         replacementSet = genActionSeqSetFromNormalDist(mu, sigma, G-K, H, (-800, 800))
    #         actionSeqSet = torch.cat((eliteActionSeqSet, replacementSet))
    # pass

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
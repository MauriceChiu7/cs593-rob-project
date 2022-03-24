import argparse
import csv
import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import torch
import time

from unitree_pybullet.a1 import loadA1

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

"""
Gets initial mean and sigma for Normal Distributions.
"""
def initialDist(jointsForceRange, numOfPlans, horiLen):
    mu = torch.zeros(horiLen, len(jointsForceRange)).flatten()
    sigma = np.pi * torch.eye(len(mu))
    dist = genActionSeqSetFromNormalDist(mu, sigma, numOfPlans, horiLen, jointsForceRange)
    mu = torch.mean(dist, dim=0)
    sigma = torch.cov(dist.T)
    if args.verbose: print(f"initial mu:\n{mu}")
    if args.verbose: print(f"initial mu.size():\n{mu.size()}")
    if args.verbose: print(f"initial sigma:\n{sigma}")
    if args.verbose: print(f"initial sigma.size():\n{sigma.size()}")
    return (mu, sigma)

"""
Generates a action sequence set
with mean of mu and stdev of sigma of size 
[numOfPlans, numOfJoints * horiLen]
"""
def genActionSeqSetFromNormalDist(mu, sigma, numOfPlans, horiLen, jointsForceRange):
    mins, maxes = np.array(jointsForceRange).T
    actionSeqSet = []
    for _ in range(numOfPlans):
        samp = np.random.multivariate_normal(mu.numpy(), sigma.numpy()).reshape(horiLen, len(mins))
        samp = np.clip(samp, mins, maxes)
        actionSeqSet.append(torch.tensor(samp.reshape(-1)))
    actionSeqSet = torch.stack(actionSeqSet)
    if args.verbose: print(f"actionSeqSet:\n{actionSeqSet}")
    if args.verbose: print(f"actionSeqSet.size():\n{actionSeqSet.size()}")
    return actionSeqSet

"""
Calculates the cost of an action sequence for the UR5 robot.
"""
def ur5_actionSeqCost():
    pass

"""
Calculates the cost of an action sequence for the A1 robot.
"""
def a1_actionSeqCost():
    pass

"""
Loads pybullet environment with a horizontal plane and earth like gravity.
"""
def loadEnv():
    if args.verbose: print(f"\nloading environment...\n")
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0.1])
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1./500)
    # p.setRealTimeSimulation(1)
    if args.verbose: print(f"\n...environment loaded\n")

"""
Loads the UR5 robot.
"""
def loadUR5(activeJoints):
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
Moves the robots to their starting position.
"""
def moveToStartingPose(uid, robot, jointIds):
    if args.robot == 'ur5':
        for _ in range(100):
            # applyAction(uid, jointIds, [-2.6,-1.5,1.7,0,0,0,0,0])
            pass
    else:
        for _ in range(50):
            p.stepSimulation()

def main():
    if args.verbose: print(f"args: {args}")
    loadEnv()

    uid = jointsForceRange = None
    END_EFFECTOR_INDEX = N = G = H = T = K = ACTIVE_JOINTS = Goal = None

    # Setting up robot specific constants.
    if args.robot == 'ur5':
        # Setting up trajectory for UR5.
        resolution = 0.1
        trajX = [-1 * (5 + 0.0 * np.cos(theta * 4)) * np.cos(theta) for theta in np.arange(-np.pi + 0.2, np.pi - 0.2, resolution)]
        trajY = [-1 * (5 + 0.0 * np.cos(theta * 4)) * np.sin(theta) for theta in np.arange(-np.pi + 0.2, np.pi - 0.2, resolution)]
        trajZ = [5 for z in np.arange(-np.pi + 0.2, np.pi - 0.2, resolution)]
        traj = np.array(list(zip(trajX, trajY, trajZ))) / 10
        if args.verbose: print(f"trajectory length: {len(traj)}")
        END_EFFECTOR_INDEX = 7 # The end effector link index.
        N = len(traj)    # Number of environmental steps.
        G = 20           # Number of plans.
        H = 5            # The horizon length.
        T = 10           # Times to update mean and standard deviation for the distribution.
        K = int(0.4 * G) # Numbers of action sequences to keep.
        if args.verbose: print(f"G = {G}, H = {H}, T = {T}, K = {K}")
        ACTIVE_JOINTS = [1,2,3,4,5,6,8,9]
        uid, jointsForceRange = loadUR5(ACTIVE_JOINTS)
    else: 
        # Setting up goal coordinates for A1.
        Goal = (100, 0, p.getLinkState(uid, 2)[0][2])
        N = 100
        G = 30
        H = 5
        T = 30
        K = int(0.4 * G)
        if args.verbose: print(f"N = {N}, G = {G}, H = {H}, T = {T}, K = {K}")
        uid, jointsForceRange, activeJoints = loadA1()
        ACTIVE_JOINTS = activeJoints

    # moveToStartingPose()

    mu, sigma = initialDist(jointsForceRange, 100, H)

    # ___LINE 0___
    bestActions = []
    for envStep in range(N):
        stateId = p.saveState() # save the state before simulation.

        # Get H future destinations from trajectory
        futureStates = [] # For the UR5 robot only.
        if args.robot == 'ur5':
            for h in range(H):
                if envStep + h > len(traj) - 1:
                    futureStates.append(traj[(envStep + h)%len(traj)])
                else:
                    futureStates.append(traj[envStep + h])

        # ___LINE 1___
        # Generates G action sequences of horizon length H.
        actionSeqSet = genActionSeqSetFromNormalDist(mu, sigma, G, H, jointsForceRange)

        # ___LINE 2___
        # Get robot's current state (not needed right now).

        # ___LINE 3___
        for t in range(T):
            if args.verbose: print(f"\n...training envStep {envStep}, iteration {t}")
        # ___LINE 4a___
        # (Milestone 3) Directly modify your action sequence using Gradient optimization. 
        # It takes your generated action sequences, cost, and "back propagation" and returns a better action sequence. 
        # Done through training a graph neural network to learn the "images" of our robots.

        # ___LINE 4b___
        # Get state sequence from action sequence.
        planCosts = []
        for actionSeq in actionSeqSet:
            p.restoreState(stateId)

            # ___LINE 5___
            # Calculate the cost of the state sequence.
            cost = 0
            if args.robot == 'ur5':
                cost = ur5_actionSeqCost()
            else:
                cost = a1_actionSeqCost()

            planCosts.append((actionSeq, cost))

        # ___LINE 6___
        # Sort action sequences by cost.
        sortedActionSeqSet = sorted(planCosts, key = lambda x: x[1])

        # ___LINE 7___
        # Update normal distribution to fit top K action sequences.
        eliteActionSeqSet = []
        for eliteActionSeq in range(K):
            eliteActionSeqSet.append(sortedActionSeqSet[eliteActionSeq][0])
        eliteActionSeqSet = torch.stack(eliteActionSeqSet)

        mu = torch.mean(eliteActionSeqSet, dim=0)
        sigma = torch.cov(eliteActionSeqSet.T)
        sigma += .02 * torch.eye(len(mu)) # add a small amount of noise to the diagonal to replan to next target

        # ___LINE 8___
        # Replace bottom G-K sequences with better action sequences.
        replacementSet = genActionSeqSetFromNormalDist(mu, sigma, G-K, H, jointsForceRange)
        actionSeqSet = torch.cat((eliteActionSeqSet, replacementSet))

        # ___LINE 9___
        # Execute the best action.
        bestAction = actionSeqSet[0][:len(ACTIVE_JOINTS)]
        # if args.verbose: print(f"\nbestAction: {bestAction}\n")
        
        p.restoreState(stateId)
        # applyAction(ACTIVE_JOINTS, bestAction)
        # bestActions.append(bestAction.tolist())

    # while 1:
    #     p.stepSimulation()
    if args.verbose: print(f"\nwriting best actions to file...\n")
    filename = f"../{args.robot}_best_actions.csv"
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(bestActions)
    if args.verbose: print(f"\n...best actions wrote to file: ./{args.robot}_best_actions.csv\n")

def playback():
    if args.verbose: print(f"\nreading best actions from file: ./{args.robot}_best_actions.csv...\n")
    filename = "./{args.robot}_best_actions.csv"
    file = open(filename)
    csvreader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
    finalActions = []
    for row in csvreader:
        finalActions.append(row)
    file.close()
    if args.verbose: print(f"\n...best actions read\n")
    if args.verbose: print(f"finalActions:\n{finalActions}")
    
    loadEnv()

    ACTIVE_JOINTS = None

    if args.robot == 'ur5':
        ACTIVE_JOINTS = [1,2,3,4,5,6,8,9]
        uid, jointsForceRange = loadUR5(ACTIVE_JOINTS)
    else: 
        uid, jointsForceRange, activeJoints = loadA1()
        ACTIVE_JOINTS = activeJoints

    # moveToStartingPose()

    for env_step in range(len(finalActions)):
        # applyAction(ACTIVE_JOINTS, finalActions[env_step])
        time.sleep(1./25.)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CS 593-ROB - Project Milestone 2')
    parser.add_argument('-p', '--play', action='store_true', help='Set true to playback the recorded best actions.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Logs debug information.')
    parser.add_argument('-r', '--robot', default='ur5', help='Choose which robot, "ur5" or "a1", to simulate or playback actions.')
    # parser.add_argument('-G', '--nplan', help='Number of plans to generate.')
    # parser.add_argument('-T', '--train', help='Number of iterations to train.')
    # parser.add_argument('-H', '--horizon', default=5, help='Set the horizon length.')
    # parser.add_argument('')

    args = parser.parse_args()
    
    if args.play:
        playback()
    else:
        main()
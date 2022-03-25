import argparse
import csv
from datetime import datetime
import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import torch
import time

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
Generates a action sequence set
with mean of mu and stdev of sigma of size 
[numOfPlans, numOfJoints * horiLen]
"""
def genActionSeqSetFromNormalDist(mu, sigma, numOfPlans, horiLen, jointsForceRange):
    mins, maxes = np.array(jointsForceRange).T
    actionSeqSet = []
    for _ in range(numOfPlans):
        samp = np.random.multivariate_normal(mu.numpy(), sigma.numpy()).reshape(horiLen, len(mins))
        # samp = np.clip(samp, mins, maxes)
        actionSeqSet.append(torch.tensor(samp.reshape(-1)))
    actionSeqSet = torch.stack(actionSeqSet)
    if args.verbose: print(f"actionSeqSet:\n{actionSeqSet}")
    if args.verbose: print(f"actionSeqSet.size():\n{actionSeqSet.size()}")
    return actionSeqSet

"""
Applys a random action to the all the joints.
"""
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

"""
Applies an action to each joint of the UR5 then returns the corresponding state.
"""
def ur5_getState(action, uid, jointIds):
    END_EFFECTOR_INDEX = 7 # The end effector link index.
    applyAction(uid, jointIds, action)
    eePos = p.getLinkState(uid, END_EFFECTOR_INDEX, 1)[0]
    return eePos

"""
Applies an action to each joint of the A1 then returns the corresponding state.
"""
def a1_getState(uid, jointIds, action):
    applyAction(uid, jointIds, action)
    floating_base_pos = p.getLinkState(uid, 0)[0] # a1's floating base center of mass position
    return floating_base_pos

def getJointsVelocity(uid, jointIds):
    linksState = p.getLinkStates(uid, jointIds, computeLinkVelocity=1)
    jointsVelocity = []
    for ls in linksState:
        jointsVelocity.append(ls[6])
    # print(f"jointsVelocity: {jointsVelocity}")
    return jointsVelocity

"""
Calculates the cost of an action sequence for the UR5 robot.
"""
def ur5_actionSeqCost(uid, jointIds, actionSeq, H, futureDests, stateId):
    p.restoreState(stateId)
    actionSeq2 = actionSeq.reshape(H, -1) # Get H action sequences
    cost = distCost = velCost = accCost = 0
    for h in range(H):
        st = ur5_getState(actionSeq2[h], uid, jointIds)
        htarg = futureDests[h]
        # END_EFFECTOR_INDEX = 7 # The end effector link index.
        # eePos = p.getLinkState(uid, END_EFFECTOR_INDEX, 1)[0]
        # print(f"\nhtarg: {htarg}")
        # print(f"eePos: {eePos}\n")
        if args.verbose: print(f"dist: {dist(st, htarg)}")
        distCost += dist(st, htarg)
    # if args.verbose: print(f"...distCost: {distCost}")

    p.restoreState(stateId)
    v0 = getJointsVelocity(uid, jointIds)
    v0 = torch.tensor(v0)
    
    for h in range(H):
        st = ur5_getState(actionSeq2[h], uid, jointIds)
        v = getJointsVelocity(uid, jointIds)
        v = torch.tensor(v)
        # print(f"v0: \n{v0}, \nv: \n{v}")
        a = torch.sub(torch.mul(v, v), torch.mul(v0, v0))
        v0 = v
        # print(f"a: \n{a}")
        velCost += torch.square(torch.sum(v))
        accCost += torch.square(torch.sum(a))

    weight = [1e2, 1e-02, 1]
    cost = weight[0] * distCost + weight[1] * accCost - weight[2] * velCost
    if args.verbose: print(f"...distCost: {weight[0] * distCost}, accCost: {weight[1] * accCost}, velCost: {weight[2] * velCost}")
    # ...distCost: 55.259173514720516, accCost: 2.9334241439482665e-20, velCost: 6.145710074179078e-08
    # return distCost
    return cost # The action sequence cost

"""
Calculates the cost of an action sequence for the A1 robot.
"""
def a1_actionSeqCost(uid, jointIds, actionSeq, H, goal):
    weights = [1, 100]
    # Reshape action sequence to array of arrays (originally just a single array)
    actionSeq = actionSeq.reshape(H, -1)
    # Initialize cost
    cost = 0
    # Loop through each action of the plan and add cost
    for h in range(H):
        currAction = actionSeq[h]
        state = a1_getState(uid, jointIds, currAction)
        distCost = weights[0] * dist(state, goal) # distance from goal
        actionCost = weights[1] * dist(actionSeq[h], torch.zeros(len(currAction))) # gets the magnitude of actions (shouldn't apply huge actions)
        cost += distCost   
        cost += actionCost 
        if args.verbose: print(f"...dist cost: {distCost}, action cost: {actionCost}")
    return cost

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
def moveToStartingPose(uid, jointIds):
    if args.robot == 'ur5':
        if args.verbose: print(f"\nmoving UR5 to starting pose...\n")
        for _ in range(160):
            applyAction(uid, jointIds, [-1600,-1500,0,0,0,0,0,0])
        if args.verbose: print(f"...UR5 moved to starting pose\n")
        
    else:
        if args.verbose: print(f"\nwaiting for A1 to settle...\n")
        for _ in range(50):
            p.stepSimulation()
        if args.verbose: print(f"A1 has settled...\n")

SIM_STEPS = 3
CTL_FREQ = 20    # Hz
LOOKAHEAD_T = 2  # s
EXEC_T = .5      # s

def main():
    startTime = datetime.now()
    startTimeStr = startTime.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\ntraing start time: {startTimeStr}\n")

    if args.verbose: print(f"args: {args}")
    loadEnv()

    uid = jointsForceRange = None
    traj = N = G = H = T = K = ACTIVE_JOINTS = goal = H_exec = None

    # Setting up robot specific constants.
    if args.robot == 'ur5':
        # Setting up trajectory for UR5.
        # resolution = 0.01
        resolution = 0.1
        trajX = [-1 * (5 + 0.0 * np.cos(theta * 4)) * np.cos(theta) for theta in np.arange(-np.pi + 0.2, np.pi - 0.2, resolution)]
        trajY = [-1 * (5 + 0.0 * np.cos(theta * 4)) * np.sin(theta) for theta in np.arange(-np.pi + 0.2, np.pi - 0.2, resolution)]
        trajZ = [5 for z in np.arange(-np.pi + 0.2, np.pi - 0.2, resolution)]
        traj = np.array(list(zip(trajX, trajY, trajZ)))/8
        if args.verbose: print(f"trajectory length: {len(traj)}")
        if args.debug:
            for pt in range(len(traj)-1):
                p.addUserDebugLine(traj[pt],[0,0,0],traj[pt+1])
        # while 1:
        #     p.stepSimulation()
        N = len(traj)                               # Number of environmental steps.
        N = 20
        G = 200                                     # Number of plans.
        H = 5
        # H = int(np.ceil(CTL_FREQ * LOOKAHEAD_T))  # The horizon length.
        # H_exec = int(np.ceil(CTL_FREQ * EXEC_T))
        T = 40                                      # Times to update mean and standard deviation for the distribution.
        K = int(0.4 * G)                            # Numbers of action sequences to keep.
        ACTIVE_JOINTS = [1,2,3,4,5,6,8,9]
        uid, jointsForceRange = loadUR5(ACTIVE_JOINTS)
    else: 
        # Setting up goal coordinates for A1.
        N = 20
        G = 200
        H = 5
        # H = int(np.ceil(CTL_FREQ * LOOKAHEAD_T))
        # H_exec = int(np.ceil(CTL_FREQ * EXEC_T))
        T = 20
        K = int(0.4 * G)
        uid, jointsForceRange, activeJoints = loadA1()
        goal = (100, 0, p.getLinkState(uid, 2)[0][2])
        if args.verbose: print(f"set A1's goal to: {goal}")
        ACTIVE_JOINTS = activeJoints
    print(f"N = {N}, G = {G}, H = {H}, H_exec = {H_exec}, T = {T}, K = {K}")
    if args.verbose: print(f"ACTIVE_JOINTS: {ACTIVE_JOINTS}")

    moveToStartingPose(uid, ACTIVE_JOINTS)
    # while 1:
    #     p.stepSimulation()

    mu = torch.zeros(H, len(jointsForceRange)).flatten()
    if args.robot == 'ur5':
        # sigma = (np.pi * 1e05) * torch.eye(len(mu))
        sigma = 2e5 * torch.eye(len(mu))
    else: 
        sigma = (np.pi * 1e06) * torch.eye(len(mu))
    if args.verbose: print(f"initial mu:\n{mu}")
    if args.verbose: print(f"initial mu.size():\n{mu.size()}")
    if args.verbose: print(f"initial sigma:\n{sigma}")
    if args.verbose: print(f"initial sigma.size():\n{sigma.size()}")

    # ___LINE 0___
    finalActions = []
    for envStep in range(N):
        if not args.verbose: print(f"\ntraining envStep {envStep}...")
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
            if args.verbose: print(f"\ntraining envStep {envStep}, iteration {t}...")
            # ___LINE 4a___
            # (Milestone 3) Directly modify your action sequence using Gradient optimization. 
            # It takes your generated action sequences, cost, and "back propagation" and returns a better action sequence. 
            # Done through training a graph neural network to learn the "images" of our robots.

            # ___LINE 4b___
            # Get state sequence from action sequence.
            planCosts = []
            cost = 0
            for actionSeq in actionSeqSet:
                p.restoreState(stateId)

                # ___LINE 5___
                # Calculate the cost of the state sequence.
                # cost = 0
                if args.robot == 'ur5':
                    cost = ur5_actionSeqCost(uid, ACTIVE_JOINTS, actionSeq, H, futureStates, stateId)
                else:
                    cost = a1_actionSeqCost(uid, ACTIVE_JOINTS, actionSeq, H, goal)
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
            # mu.add(cost)
            sigma = torch.cov(eliteActionSeqSet.T)
            sigma += .02 * torch.eye(len(mu)) # add a small amount of noise to the diagonal to replan to next target
            if args.verbose: print(f"mu for envStep {envStep}:\n{mu}")
            if args.verbose: print(f"sigma for envStep {envStep}:\n{sigma}")

            # ___LINE 8___
            # Replace bottom G-K sequences with better action sequences.
            replacementSet = genActionSeqSetFromNormalDist(mu, sigma, G-K, H, jointsForceRange)
            actionSeqSet = torch.cat((eliteActionSeqSet, replacementSet))

        # ___LINE 9___
        # Execute the best action.
        # bestActions = [actSeq[:len(ACTIVE_JOINTS)] for i, actSeq in enumerate(actionSeqSet) if i < H_exec]
        bestAction = actionSeqSet[0][:len(ACTIVE_JOINTS)]
        p.restoreState(stateId)
        applyAction(uid, ACTIVE_JOINTS, bestAction)
        finalActions.append(bestAction.tolist())    # Keep track of all actions
        # for act in bestActions:
        #     applyAction(uid, ACTIVE_JOINTS, act)
        #     finalActions.append(act.tolist())    # Keep track of all actions

    if args.verbose: print(f"\n=== finalActions ===\n{finalActions}\n")
    
    print("training done!\n")

    # while 1:
    #     p.stepSimulation()
    endTime = datetime.now()
    endTimeStr = endTime.strftime('%Y-%m-%d %H:%M:%S')

    if args.verbose: print(f"\nwriting final actions to file...\n")
    filenameLastRun = f"./{args.robot}_final_actions.csv"
    filenameBackUp = f"./{args.robot}_final_actions_{endTimeStr}.csv"
    with open(filenameLastRun, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(finalActions)
    if args.verbose: print(f"\n...final actions wrote to file: ./{args.robot}_final_actions.csv\n")
    with open(filenameBackUp, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(finalActions)
    if args.verbose: print(f"\n...final actions wrote to file: ./{args.robot}_final_actions_{endTimeStr}.csv\n")

    print(f"\ntraing end time: {endTimeStr}\n")
    print(f"\ntime elapsed: {endTime-startTime}\n")
    

def playback():
    if args.verbose: print(f"\nreading final actions from file: ./{args.robot}_final_actions.csv...\n")
    filename = f"./{args.robot}_final_actions.csv"
    file = open(filename)
    csvreader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
    finalActions = []
    for row in csvreader:
        finalActions.append(row)
    file.close()
    if args.verbose: print(f"\n...final actions read\n")
    if args.verbose: print(f"\n=== finalActions ===\n{finalActions}\n")
    
    loadEnv()

    ACTIVE_JOINTS = None

    if args.robot == 'ur5':
        ACTIVE_JOINTS = [1,2,3,4,5,6,8,9]
        uid, jointsForceRange = loadUR5(ACTIVE_JOINTS)
    else: 
        uid, jointsForceRange, activeJoints = loadA1()
        ACTIVE_JOINTS = activeJoints

    moveToStartingPose(uid, ACTIVE_JOINTS)

    for env_step in range(len(finalActions)):
        applyAction(uid, ACTIVE_JOINTS, finalActions[env_step])
        time.sleep(1./1.)

def test():
    print("======= in test =======")
    ACTIVE_JOINTS = [1,2,3,4,5,6,8,9]
    loadEnv()
    uid, jointsForceRange = loadUR5(ACTIVE_JOINTS)
    maxForces = []
    for f in jointsForceRange:
        # print(f)
        maxForces.append(f[1])
    while 1:
        applyAction(uid, ACTIVE_JOINTS, torch.mul(torch.tensor(maxForces), 20))

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
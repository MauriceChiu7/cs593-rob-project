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
Gets the upper and lower positional limits of each joint.
"""
def getJointsRange(uid, jointIds):
    jointsRange = []
    for a in jointIds:
        jointInfo = p.getJointInfo(uid, a)
        jointsRange.append((jointInfo[8], jointInfo[9]))
    return jointsRange

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
    # action = torch.tensor(action)
    # torqueScalar = 1
    # if args.robot == 'ur5':
    #     # correction = 105
    #     # action = torch.where(action > 0, torch.add(action, correction), torch.add(action, -correction))
    #     # torqueScalar = 15
    #     torqueScalar = 1
    # else:
    #     # torqueScalar = 15
    #     torqueScalar = 1
    # action = torch.mul(action, torqueScalar)
    # p.setJointMotorControlArray(uid, jointIds, p.TORQUE_CONTROL, forces=action)
    p.setJointMotorControlArray(uid, jointIds, p.POSITION_CONTROL, targetPositions=action)
    if args.verbose: print(f"action applied: \n{action}")
    for _ in range(SIM_STEPS):
        p.stepSimulation()

def getConfig(uid, jointIds):
    jointPositions = []
    for id in jointIds:
        # print(p.getJointState(uid, id)[0])
        jointPositions.append(p.getJointState(uid, id)[0])
    jointPositions = torch.Tensor(jointPositions)
    return jointPositions

"""
Applies an action to each joint of the UR5 then returns the corresponding state.
"""
def ur5_getState(action, uid, jointIds):
    applyAction(uid, jointIds, action)
    eePos = p.getLinkState(uid, END_EFFECTOR_INDEX, 1)[0]
    return eePos

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
    cost = distCost = actionCost = velCost = accCost = 0
    for h in range(H):
        curr_elbow = p.getLinkState(uid, 3)[0]
        st = ur5_getState(actionSeq2[h], uid, jointIds)
        htarg = futureDests[h]
        # END_EFFECTOR_INDEX = 7 # The end effector link index.
        # eePos = p.getLinkState(uid, END_EFFECTOR_INDEX, 1)[0]
        # print(f"\nhtarg: {htarg}")
        # print(f"eePos: {eePos}\n")
        next_elbow = p.getLinkState(uid, 3)[0]
        distCost = 1 * dist(st, htarg)
        actionCost = 1 * dist(next_elbow, curr_elbow)

        # print(f"distCost: {distCost}")
        # print(f"actionCost: {actionCost}")

        cost = cost + distCost + actionCost
        # if args.verbose: print(f"dist: {dist(st, htarg)}")
    # if args.verbose: print(f"...distCost: {distCost}")

    # p.restoreState(stateId)
    # v0 = getJointsVelocity(uid, jointIds)
    # v0 = torch.tensor(v0)
    
    # for h in range(H):
    #     st = ur5_getState(actionSeq2[h], uid, jointIds)
    #     v = getJointsVelocity(uid, jointIds)
    #     v = torch.tensor(v)
    #     # print(f"v0: \n{v0}, \nv: \n{v}")
    #     a = torch.sub(torch.mul(v, v), torch.mul(v0, v0))
    #     v0 = v
    #     # print(f"a: \n{a}")
    #     velCost += torch.square(torch.sum(v))
    #     accCost += torch.square(torch.sum(a))

    # weight = [1e2, 1e-02, 1]
    # cost = weight[0] * distCost + weight[1] * accCost - weight[2] * velCost
    # if args.verbose: print(f"...distCost: {weight[0] * distCost}, accCost: {weight[1] * accCost}, velCost: {weight[2] * velCost}")
    # ...distCost: 55.259173514720516, accCost: 2.9334241439482665e-20, velCost: 6.145710074179078e-08
    # return distCost
    return cost # The action sequence cost

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
    if args.verbose: print(f"\nloading UR5...\n")
    p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=(0,0,0))
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

SIM_STEPS = 3
CTL_FREQ = 20    # Hz
LOOKAHEAD_T = 2  # s
EXEC_T = .5      # s
END_EFFECTOR_INDEX = 7 # The end effector link index.
ACTIVE_JOINTS = [1,2,3,4,5,6,8,9]
DISCRETIZED_STEP = 0.005

def main():
    startTime = datetime.now()
    startTimeStr = startTime.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\ntraing start time: {startTimeStr}\n")

    if args.verbose: print(f"args: {args}")
    loadEnv()

    uid = jointsForceRange = jointsRange= None
    traj = N = G = H = T = K = None

    # Generate random goal state for UR5
    x = np.random.uniform(-0.7, 0.7)
    y = np.random.uniform(-0.7, 0.7)
    z = np.random.uniform(0.1, 0.7)
    goalState = torch.Tensor([x, y, z])
    p.addUserDebugLine([0,0,0.1], goalState, [0,0,1])
    print(f"\ngoalState: {goalState}\n")

    
    
    uid, jointsForceRange = loadUR5(ACTIVE_JOINTS)
    jointsRange = getJointsRange(uid, ACTIVE_JOINTS)
    if args.verbose: print(f"ACTIVE_JOINTS: {ACTIVE_JOINTS}")
    

    # Start ur5 with random positions
    jointsRange = getJointsRange(uid, ACTIVE_JOINTS)
    random_positions = []
    for r in jointsRange:
        rand = np.random.uniform(r[0], r[1])
        random_positions.append(rand)
    # print(jointsRange)
    print(f"random_positions: {random_positions}")
    p.setJointMotorControlArray(uid, ACTIVE_JOINTS, p.POSITION_CONTROL, random_positions)
    # Give it some time to move there
    for _ in range(100):
        p.stepSimulation()
    s0 = getConfig(uid, ACTIVE_JOINTS)
    print(f"s0: {s0}")
    startState = p.getLinkState(uid, END_EFFECTOR_INDEX, 1)[0]
    p.addUserDebugLine([0,0,0.1], startState, [1,0,0])

    distTotal = dist(goalState, startState)
    diffBtwin = np.array(diff(goalState, startState))
    incrementTotal = distTotal/DISCRETIZED_STEP
    numSegments = int(math.floor(incrementTotal))+1
    stepVector = diffBtwin / numSegments

    print(startState)
    print(goalState)
    # print("===========")
    # print(distTotal)
    # print(diffBtwin)
    # print(incrementTotal)
    # print(numSegments)
    # print(stepVector)
    
    traj = []
    traj.append(np.array(startState))
    state = np.copy(startState)
    for _ in range(numSegments):
        state += stepVector
        traj.append(np.copy(state))

    print(traj)

    # exit()

    N = numSegments                             # Number of environmental steps.
    G = 220                                     # Number of plans.
    G = 200
    H = 20    
    T = 120                                     # Times to update the distribution.
    T = 80
    K = int(0.3 * G)                            # Numbers of action sequences to keep.
    print(f"\nN = {N}, G = {G}, H = {H}, T = {T}, K = {K}")
    
    # ___LINE 0___
    finalActions = []
    for envStep in range(N):
        # if not args.verbose: print(f"\ntraining envStep {envStep}/{N-1}...")
        stateId = p.saveState() # save the state before simulation.

        # Get H future destinations from trajectory
        futureStates = [] # For the UR5 robot only.
        
        for h in range(H):
            if envStep + h > len(traj) - 1:
                futureStates.append(traj[(envStep + h)%len(traj)])
            else:
                futureStates.append(traj[envStep + h])

        mu = torch.zeros(H, len(jointsRange)).flatten()
        
        # sigma = (np.pi * 100) * torch.eye(len(mu))
        sigma = (np.pi**2) * torch.eye(len(mu))
        # sigma = 2e5 * torch.eye(len(mu))
        
        if args.verbose: print(f"initial mu:\n{mu}")
        if args.verbose: print(f"initial mu.size():\n{mu.size()}")
        if args.verbose: print(f"initial sigma:\n{sigma}")
        if args.verbose: print(f"initial sigma.size():\n{sigma.size()}")

        # ___LINE 1___
        # Generates G action sequences of horizon length H.
        actionSeqSet = genActionSeqSetFromNormalDist(mu, sigma, G, H, jointsRange)

        # ___LINE 2___
        # Get robot's current state (not needed right now).

        # ___LINE 3___
        for t in range(T):
            print(f"training envStep {envStep}/{N-1}, iteration {t}/{T-1}...")
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
                cost = ur5_actionSeqCost(uid, ACTIVE_JOINTS, actionSeq, H, futureStates, stateId)
                
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
            replacementSet = genActionSeqSetFromNormalDist(mu, sigma, G-K, H, jointsRange)
            actionSeqSet = torch.cat((eliteActionSeqSet, replacementSet))

        # ___LINE 9___
        # Execute the best action.
        # bestActions = [actSeq[:len(ACTIVE_JOINTS)] for i, actSeq in enumerate(actionSeqSet) if i < H_exec]
        bestAction = actionSeqSet[0][:len(ACTIVE_JOINTS)]
        p.restoreState(stateId)
        # if args.verbose: print(f"\n===== bestAction =====")
        applyAction(uid, ACTIVE_JOINTS, bestAction)
        # if args.verbose: print(f"======================\n")
        finalActions.append(bestAction.tolist())    # Keep track of one best action
        # for act in bestActions:                   # Keep track of all best actions
        #     applyAction(uid, ACTIVE_JOINTS, act)
        #     finalActions.append(act.tolist())

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

    goalState = [ 0.1988, -0.0352,  0.6358]
    p.addUserDebugLine([0,0,0.1], goalState, [0,0,1])
    s0 = [ 1.4018, -0.5394, -0.5875, -0.8265, -3.1231, -1.2301,  0.0077, -0.0093]

    ACTIVE_JOINTS = [1,2,3,4,5,6,8,9]
    uid, jointsForceRange = loadUR5(ACTIVE_JOINTS)
    

    # Start ur5 with random positions
    # jointsRange = getJointsRange(uid, ACTIVE_JOINTS)
    # random_positions = []
    # for r in jointsRange:
    #     rand = np.random.uniform(r[0], r[1])
    #     random_positions.append(rand)
    # # print(jointsRange)
    # print(f"random_positions: {random_positions}")
    p.setJointMotorControlArray(uid, ACTIVE_JOINTS, p.POSITION_CONTROL, s0)
    # Give it some time to move there
    for _ in range(100):
        p.stepSimulation()

    p.addUserDebugLine([0,0,0.1], p.getLinkState(uid, END_EFFECTOR_INDEX, 1)[0], [1,0,0])


    for env_step in range(len(finalActions)):
        time.sleep(1.)
        applyAction(uid, ACTIVE_JOINTS, finalActions[env_step])
        # time.sleep(1./1.)

# def test():
#     print("======= in test =======")
#     ACTIVE_JOINTS = [1,2,3,4,5,6,8,9]
#     loadEnv()
#     uid, jointsForceRange = loadUR5(ACTIVE_JOINTS)
#     maxForces = []
#     for f in jointsForceRange:
#         # print(f)
#         maxForces.append(f[1])
#     while 1:
#         applyAction(uid, ACTIVE_JOINTS, torch.mul(torch.tensor(maxForces), 20))

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
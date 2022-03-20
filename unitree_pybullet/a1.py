import argparse
from random import getstate
from tkinter.ttk import Separator
from turtle import right
import pybullet as p
import os
import pybullet_data
import numpy as np
import torch
import csv
import math
import time

maxForceId = 0

def diff(v1, v2):
    return [x1 - x2 for x1, x2 in zip(v1, v2)]

def magnitude(v):
    return math.sqrt(sum([x*x for x in v]))

def dist(p1, p2):
    return magnitude(diff(p1, p2))

def loadA1():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), 0, 0, 0)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1./500)
    # p.setRealTimeSimulation(1)

    # urdfFlags = p.URDF_USE_SELF_COLLISION
    urdfFlags = p.URDF_USE_INERTIA_FROM_FILE
    # quadruped = p.loadURDF("data/a1/urdf/a1.urdf",[0,0,0.48],[0,0,0,1], flags = urdfFlags,useFixedBase=False)
    quadruped = p.loadURDF("data/a1/urdf/a1.urdf",[0,0,0.48], p.getQuaternionFromEuler([0,0,0]), flags=urdfFlags)

    # getJointsInfo(p, quadruped, True)
    # exit(0)

    #enable collision between lower legs
    lower_legs = [2,5,8,11]
    for l0 in lower_legs:
        for l1 in lower_legs:
            if (l1>l0):
                enableCollision = 1
                # print("collision for pair",l0,l1, p.getJointInfo(quadruped,l0)[12],p.getJointInfo(quadruped,l1)[12], "enabled=",enableCollision)
                p.setCollisionFilterPair(quadruped, quadruped, 2, 5, enableCollision)

    jointIds=[]

    maxForceId = p.addUserDebugParameter("maxForce",0,500,20)

    # Set resistance to none
    for j in range(p.getNumJoints(quadruped)):
        p.changeDynamics(quadruped, j, linearDamping=0, angularDamping=0)
        info = p.getJointInfo(quadruped,j)
        jointName = info[1]
        jointType = info[2]
        if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
            jointIds.append(j)

    # Adding extra lateral friction to the feet
    foot_fixed = [5, 9, 13, 17] # 5: FntRgt, 9: FntLft, 13: RarRgt, 17: RarLft
    for foot in foot_fixed:
        p.changeDynamics(quadruped, foot, lateralFriction=1)
    
    # for _ in range(50):
    #     p.stepSimulation()
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

def getJointsRange(uid, jointIds):
    jointsRange = []
    for a in jointIds:
        jointInfo = p.getJointInfo(uid, a)
        jointsRange.append((jointInfo[8], jointInfo[9]))
    return jointsRange

def getJointsMaxForce(uid, jointIds):
    jointsMaxForces = []
    for j in jointIds:
        jointInfo = p.getJointInfo(uid, j)
        jointsMaxForces.append(jointInfo[10])
    return jointsMaxForces

def simulate():
    # setup for simulation
    p.getCameraImage(480,320)
    p.setRealTimeSimulation(0)

    while(1):
        p.stepSimulation()

"""Apply a random action to the all the links/joints of the hip."""
def applyAction(quadruped, jointIds, action):
    # Makes the action read from the toggle bar for forces
    fs = [p.readUserDebugParameter(maxForceId)] * 12 # Feel like this should be proportional to each joint and not the same for every joint. 
    # p.setJointMotorControlArray(quadruped, jointIds, p.POSITION_CONTROL, action, forces=fs)
    p.setJointMotorControlArray(quadruped, jointIds, p.POSITION_CONTROL, action)
    for _ in range(10):
        p.stepSimulation()

def getState(quadruped, jointIds, action):    # Changed
    applyAction(quadruped, jointIds, action)
    floating_base_pos = p.getLinkState(quadruped, 0)[0] # a1's floating base center of mass position
    return floating_base_pos

"""Get x,y,z coordinates of the hip joints."""
# def getState(quadruped):
#     FR_hip_joint = p.getLinkState(quadruped, 2)[0]
#     FL_hip_joint = p.getLinkState(quadruped, 6)[0]
#     RR_hip_joint = p.getLinkState(quadruped, 10)[0]
#     RL_hip_joint = p.getLinkState(quadruped, 14)[0]
#     return [FR_hip_joint, FL_hip_joint, RR_hip_joint, RL_hip_joint]

# """Calculates the total cost of each path/plan."""
# def getPathCost(stateSet, s0):
#     cost = 0
#     for x in range(len(stateSet)):
#         if x == 0:
#             cost += getStateCost(stateSet[x], s0, s0)
#         else:
#             cost += getStateCost(stateSet[x], stateSet[x-1], s0)
#     return cost

"""Calculates the total cost of each path/plan."""
def getPathCost(quadruped, jointIds, actionSeq, s0, H):
    resolution = 0.5
    hFutureStates = []
    x, y, z = s0
    for h in range(H):
        x += resolution
        hFutureStates.append((x, y, z))
    actionSeq = actionSeq.reshape(H, -1)
    cost = 0
    state = getState(quadruped, jointIds, actionSeq[0])
    target = hFutureStates[0]
    cost += dist(state, target)
    
    return cost

"""Calculate the cost of moving to the 'next' state."""
def getStateCost(state, prevState, s0):
    z_init = 0.480031    # Z-coord of hip joint at initial state
    # z_prev = (prevState[0][2] + prevState[1][2] + prevState[3][2] + prevState[4][2])/4 
    z1 = state[0][2]
    z2 = state[1][2]
    z3 = state[2][2]
    z4 = state[3][2]

    ##### Calculating Tilt of Floating Body
    sums = 0
    sums += abs(z1 - z2)
    sums += abs(z1 - z3)
    sums += abs(z1 - z4)
    sums += abs(z2 - z3)
    sums += abs(z2 - z4)
    sums += abs(z3 - z4)
    body_tilt = sums/6   # get average
    #####

    # Calculating how high in air
    # Just want z-coordinate to stay same--not in air or crouching
    avgZ = (z1+z2+z3+z4)/4
    heightDiff = abs(avgZ - z_init)

    ##### Calculate FR and FL hip's x,y coordinates
    # calculate where the FR and FL hip's x coordinates are
    currX = (state[0][0] + state[1][0]) * 0.5
    prevX = (prevState[0][0] + prevState[1][0]) * 0.5
    diffX = prevX - currX   # Calculate if moving forward

    # calculate where the FR and FL hip's y coordinates are
    diffY = (abs(state[0][1] + state[1][1]) + abs(state[2][1] + state[3][1])) * 0.5
    # diffY = currY    # Calculate if straight
    #####

    ##### Calculate cost from target_x
    # Want it to move 5 units in x direction
    target_x = []
    for tup in s0:
        t = (tup[0] + 5, tup[1], tup[2])
        target_x.append(t)

    dist_cost = 0
    for n in range(4):
        dist_cost += dist(list(state[n]), list(s0[n]))

    # print(s0)
    # print(state)
    # print(dist_cost)

    # totalCost = 10*body_tilt + 10*heightDiff + 20*diffX
    # totalCost = 100*body_tilt + 100*heightDiff + 100*diffX + 30*diffY + foot_diff*100
    # totalCost = 100*body_tilt + 1000*heightDiff + 100*diffX + 100*diffY + foot_diff*100
    totalCost = 30*body_tilt + 10*heightDiff + 40*diffX + 30*diffY
    return totalCost

def sampleNormDistr(jointsRange, mu, sigma, G, H):
    mins, maxes = np.array(jointsRange).T
    actionSeqSet = [] # action sequence set that contains g sequences
    for g in range(G):
        samp = np.random.multivariate_normal(mu.numpy(), sigma.numpy()).reshape(H, len(mins))
        samp = np.clip(samp, mins, maxes)
        actionSeqSet.append(torch.tensor(samp.reshape(-1)))
    return torch.stack(actionSeqSet)

# def sampleNormDistr(jointIds, mu, sigma, G, H):
#     # maxes = [0.23,0.81,-0.98,0.14,0.83,-0.95,0.28,1.32,-0.94,0.13,1.30,-0.93]
#     # maxes = [1, 3.2, 0.5, 0.6, 3.2, 0.5, 1.2, 5.2, 0.5 ,0.6, 5.2, 0.5]
#     maxes = [3, 3.2, 0.5, 3, 3.2, 0.5]
#     maxes = torch.tensor(maxes)
#     # mins = [-0.15, 0.63, -1.75, -0.27, 0.53, -1.74, -0.13, 0.27, -1.84, -0.25, 0.24, -1.84]
#     # mins = [-0.6, -0.5, -7.0, -1.0, -0.5, -7.0, -0.6, -0.5, -7.2, -1.0, -0.5, -7.2]
#     mins = [1, -0.5, -7.0, 1, -0.5, -7.0]
#     mins = torch.tensor(mins)

#     # action_sequence = []
#     # samp = np.random.multivariate_normal(mu.numpy(), sigma.numpy()).reshape(H, len(mins))
#     # samp = np.clip(samp, mins, maxes)
#     # action_sequence.append(torch.tensor(samp.reshape(-1)))
#     # for x in range(len(mu)):
#     #     action = torch.normal(mean = mu[x], std = sigma[x])
#     #     action = torch.clamp(action, mins, maxes)
#     #     action = action.tolist()

#     #     # This is needed if you have mu vector of size 6
#     #     front_right = action[:3]
#     #     front_left = action[3:]
#     #     action = action + front_left + front_right
#     #     action_sequence.append(action)
#     actionSeqSet = [] # action sequence set that contains g sequences
#     for _ in range(G):
#         samp = np.random.multivariate_normal(mu.numpy(), sigma.numpy()).reshape(H, len(mins))
#         samp = np.clip(samp, mins, maxes)
#         actionSeqSet.append(torch.tensor(samp.reshape(-1)))

#     return torch.stack(actionSeqSet)


def getStateSeq(action_sequence, H, quadruped, jointIds, currentID):
    action_sequence = action_sequence.reshape(H, -1)
    state_sequence = []
    for h in range(H):
        temp = torch.cat((action_sequence[h], action_sequence[h][3:]))
        full = torch.cat((temp, action_sequence[h][:3]))
        applyAction(quadruped, jointIds, full)
        s = getState(quadruped)
        state_sequence.append(s)
    
    return state_sequence


def main():
    urdfFlags, quadruped = loadA1()
    envInfo = getEnvInfo(quadruped)
    jointInfo, jointStates = getRobotInfo(quadruped)
    finalActions = []
    jointIds = [2,3,4,6,7,8,10,11,12,14,15,16]   # all joints excluding foot, body, imu_joint
    jointsRange = getJointsRange(quadruped, jointIds)

    ####### Milestone 2 ######

    # Initial Variables
    G = 10                  # G is the number of paths generated (with the best 1 being picked)
    H = 5                   # Number of states to predict per path (prediction horizon)
    N = 100                 # How many iterations we're running the training for
    K = int(0.4 * G)        # Choosing the top k paths to create new distribution
    T = 5                   # Number of training iteration

    # mu = torch.tensor([[0.]*12]*H)
    # sigma = torch.tensor([[0.2]*12]*H)
    # sigma = torch.tensor([[0.2]*6]*H)

    # mu = torch.tensor([[0.]*6]*H).flatten()
    mu = torch.zeros(H, len(jointIds)).flatten() # Changed
    sigma = np.pi * torch.eye(len(mu))
    
    # print("1\n", sigma)
    count = 0

    # ________________LINE 0________________
    for _ in range(N):
        currentID = p.saveState() # Save the state before simulations. # Changed

        # ________________LINE 1________________
        # Sample G initial plans
        # Generate all the random actions for each plan
        # for _ in range(G):
        #     plans.append(sampleNormDistr(jointIds, mu, sigma, G, H))
        plans = sampleNormDistr(jointsRange, mu, sigma, G, H)
        # print(plans)
        # exit(0)

        # ________________LINE 2________________
        # Get initial states of the joints
        # Just using floating base state for now
        s0 = p.getLinkState(quadruped, 0)[0] # a1's floating base center of mass position

        # ________________LINE 3________________
        for _ in range(T):
            print("Start: ", p.getLinkState(quadruped, 0)[0]) # a1's floating base center of mass position

            # ________________LINE 4b________________
            # get sequence states from sampled sequence of actions
            # initialize path set and fill it with G paths generated from the actions
            # pathSet = []
            # for g in range(G):                            # Changed
            #     state_sequence = getStateSeq(plans[g], H, quadruped, jointIds, currentID)
            #     pathSet.append((state_sequence, plans[g]))

            # ________________LINE 5________________
            # get cost of each path (sequence of states) generated above
            actionSetCosts = []
            # for path in pathSet:                          # Changed
            #     cost = getPathCost(path[0], s0)
            #     actionSetCosts.append((path[1],cost))
            for plan in plans: 
                # print(plan)
                # exit(0)
                # Restore back to original state to run the plan again
                p.restoreState(currentID)
                cost = getPathCost(quadruped, jointIds, plan, s0, H)
                actionSetCosts.append((plan, cost))

            # ________________LINE 6________________
            # Sort the sequence of actions based on their cost
            sortedList = sorted(actionSetCosts, key = lambda x: x[1])
            # print("sortedList: \n", sortedList)
            
            # ________________LINE 7________________
            # Update normal distribution to fit top k action sequences

            # Take top K paths according to cost
            # topKPath = [np.array(i[0]) for i in sortedList[0:K]]

            # Each entry is all of one action, i.e. a_0
            sep_actions = []
            for j in range(K):
                sep_actions.append(sortedList[j][0])
            sta = torch.stack(sep_actions)

            mu = torch.mean(sta, dim=0)
            sigma = torch.cov(sta.T)
            sigma += .02 * torch.eye(len(mu))
            # print("2\n", sigma)

            # ________________LINE 8________________
            # Replace bottom G-K action sequences with samples from
            # bottomActions = []
            actions = sampleNormDistr(jointsRange, mu, sigma, G-K, H)
            actionSeqSet = torch.cat((sta, actions))
            # for _ in range(G-k):
            #     actions = sampleNormDistr(jointIds, mu, sigma, G, H)
            #     path = getStateSeq(actions, H, quadruped, jointIds, currentID)
            #     cost = getPathCost(path, s0)
            #     bottomActions.append((actions, cost))
            
            # topActions = sortedList[0:k]
            # actionSet = bottomActions + topActions
            # actionSet = sorted(actionSet, key = lambda x: x[1])

        # ________________LINE 9________________
        # execute first action from the "best" model in the action sorted action sequences 
        # execute action
        # six = actionSeqSet[0][:6]
        bestAction = actionSeqSet[0][:len(jointIds)]
        # [0,1,2,3,4,5]
        # 0 1 2 front right
        # 3 4 5 front left
        # 3 4 5 rear right
        # 0 1 2 rear left
        # temp = torch.cat((six, six[3:]))
        # bestAction = torch.cat((temp, six[:3]))
        p.restoreState(currentID)
        applyAction(quadruped, jointIds, bestAction)
        finalActions.append(bestAction.tolist())

        print("End: ", p.getLinkState(quadruped, 0)[0])
        # save state after
        # currentID = p.saveState()


        print("Iteration: ", count)
        print("This is the ID: ", currentID)
        count += 1


    # Write to final
    filename = "final_actions.csv"

    # writing to csv file
    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        # writing the data rows
        csvwriter.writerows(finalActions)

    print("Done Training")


def playback():
    
    filename = "./final_actions.csv"
    file = open(filename)
    csvreader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
    finalActions = []
    for row in csvreader:
        finalActions.append(row)
    # print(finalActions)
    file.close()

    flag, quadruped = loadA1()

    p.setRealTimeSimulation(0)
    p.setTimeStep(1./500)
    # p.getCameraImage(10,10)

    jointIds = [2,3,4,6,7,8,10,11,12,14,15,16]   # all joints excluding foot, body, imu_joint

    for env_step in range(len(finalActions)):
        a1Pos = p.getLinkState(quadruped, 0)[0]
        p.resetDebugVisualizerCamera( cameraDistance=2, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=a1Pos)
        applyAction(quadruped, jointIds, finalActions[env_step])
        time.sleep(1./125.)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CS 593-ROB - Project Milestone 2')
    parser.add_argument('-p', '--play', action='store_true', help='Set true to playback the recorded best actions.')

    args = parser.parse_args()
    
    if args.play:
        playback()
    else: 
        main()

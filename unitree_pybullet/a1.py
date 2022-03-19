from mimetypes import init
import pybullet as p
import time
import os
import sys
import pybullet_data
import numpy as np
import torch
import csv
import math

maxForceId = 0

def loadA1():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), 0, 0, 0)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1./500)

    urdfFlags = p.URDF_USE_SELF_COLLISION
    quadruped = p.loadURDF("data/a1/urdf/a1.urdf",[0,0,0.48],[0,0,0,1], flags = urdfFlags,useFixedBase=False)

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

    for _ in range(50):
        p.stepSimulation()

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


"""Get x,y,z coordinates of the hip joints."""
def getState(quadruped):
    FR_hip_joint = p.getLinkState(quadruped, 2)[0]
    FL_hip_joint = p.getLinkState(quadruped, 6)[0]
    RR_hip_joint = p.getLinkState(quadruped, 10)[0]
    RL_hip_joint = p.getLinkState(quadruped, 14)[0]

    # Foot states
    # FR_foot = p.getLinkState(quadruped, 5)[0]
    # FL_foot = p.getLinkState(quadruped, 9)[0]
    # RR_foot = p.getLinkState(quadruped, 13)[0]
    # RL_foot = p.getLinkState(quadruped, 17)[0]
    # return [FR_hip_joint, FL_hip_joint, RR_hip_joint, RL_hip_joint, FR_foot, FL_foot, RR_foot, RL_foot]

    # print('These are the FR, FL, RR, RL hip joints respectively: ', FR_hip_joint, FL_hip_joint, RR_hip_joint, RL_hip_joint)
    return [FR_hip_joint, FL_hip_joint, RR_hip_joint, RL_hip_joint]


"""Apply a random action to the all the links/joints of the hip."""
def applyAction(quadruped, jointIds, action):
    # Makes the action read from the toggle bar for forces
    # fs = [p.readUserDebugParameter(maxForceId)] * 12
    # p.setJointMotorControlArray(quadruped, jointIds, p.POSITION_CONTROL, action, forces=fs)

    # Generate states for each action in each plan
    p.setJointMotorControlArray(quadruped, jointIds, p.POSITION_CONTROL, action)
    # print("Action: \n", plans[g][h])
    for _ in range(10):
        p.stepSimulation()


"""Calculate the cost of moving to the 'next' state."""
def getStateCost(state, prevState):
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
    currY = (abs(state[0][1] + state[1][1]) + abs(state[2][1] + state[3][1])) * 0.5
    straightY = 0
    diffY = currY    # Calculate if straight
    #####

    ##### Calculate cost of feeting staying on ground
    # fr_footz = state[4][2]
    # fl_footz = state[5][2]
    # rr_footz = state[6][2]
    # rl_footz = state[7][2]

    # total = 0
    # total += abs(fr_footz - fl_footz)
    # total += abs(fr_footz - rr_footz)
    # total += abs(fr_footz - rl_footz)
    # total += abs(fl_footz - rr_footz)
    # total += abs(fl_footz - rl_footz)
    # total += abs(rl_footz - rr_footz)
    # foot_diff = total/6
    #####

    # totalCost = 10*body_tilt + 10*heightDiff + 20*diffX
    # totalCost = 100*body_tilt + 100*heightDiff + 100*diffX + 30*diffY + foot_diff*100
    # totalCost = 100*body_tilt + 1000*heightDiff + 100*diffX + 100*diffY + foot_diff*100
    totalCost = 30*body_tilt + 10*heightDiff + 40*diffX + 30*diffY
    return totalCost


"""Calculates the total cost of each path/plan."""
def getPathCost(stateSet, s0):
    cost = 0
    for x in range(len(stateSet)):
        if x == 0:
            cost += getStateCost(stateSet[x], s0)
        else:
            cost += getStateCost(stateSet[x], stateSet[x-1])
    return cost


def sampleNormDistr(quadruped, mu, sigma):
    maxes = [0.23,0.81,-0.98,0.14,0.83,-0.95,0.28,1.32,-0.94,0.13,1.30,-0.93]
    maxes = [1, 3.2, 0.5, 0.6, 3.2, 0.5, 1.2, 5.2, 0.5 ,0.6, 5.2, 0.5]
    maxes = [1, 3.2, 0.5, 0.6, 3.2, 0.5]
    maxes = torch.tensor(maxes)
    mins = [-0.15, 0.63, -1.75, -0.27, 0.53, -1.74, -0.13, 0.27, -1.84, -0.25, 0.24, -1.84]
    mins = [-0.6, -0.5, -7.0, -1.0, -0.5, -7.0, -0.6, -0.5, -7.2, -1.0, -0.5, -7.2]
    mins = [-0.6, -0.5, -7.0, -1.0, -0.5, -7.0]
    mins = torch.tensor(mins)

    # Need to calculate inverseKinematics
    footIDs = [5, 9, 13, 17]

    action_sequence = []
    for x in range(len(mu)):
        action = torch.normal(mean = mu[x], std = sigma[x])
        maxCmpr = torch.gt(action, maxes)
        minCmpr = torch.gt(mins, action)
        for b in range(len(maxCmpr)):
            if maxCmpr[b]:
                action[b] = maxes[b]
            if minCmpr[b]:
                action[b] = mins[b]
        action = action.tolist()

        # This is needed if you have mu vector of size 6
        front_right = action[:3]
        front_left = action[3:]
        action = action + front_left + front_right


        # Calc inverse kinematics and divide into each foot position
        targetPositions = []
        targetPositions.append((action[0:3]))
        targetPositions.append((action[3:6]))
        targetPositions.append((action[6:9]))
        targetPositions.append((action[9:12]))
        # print(targetPositions)

        f = p.calculateInverseKinematics2(quadruped, footIDs, targetPositions)
        action_sequence.append(f)

    return action_sequence
    # action_sequence = torch.normal(mean=mu, std=sigma)
    # return action_sequence


def getStateSeq(action_sequence, H, quadruped, jointIds, currentID):
    state_sequence = []
    for h in range(H):
        applyAction(quadruped, jointIds, action_sequence[h])
        s = getState(quadruped)
        state_sequence.append(s)
    # Restore back to original state to run the plan again
    p.restoreState(currentID)
    return state_sequence


def main():
    urdfFlags, quadruped = loadA1()
    envInfo = getEnvInfo(quadruped)
    jointInfo, jointStates = getRobotInfo(quadruped)
    finalActions = []

    ####### Milestone 2 ######

    # Initial Variables
    G = 30                 # G is the number of paths generated (with the best 1 being picked)
    H = 20                  # Number of states to predict per path (prediction horizon)
    N = 5                 # How many iterations we're running the training for
    k = int(0.4 * G)        # Choosing the top k paths to create new distribution
    currentID = p.saveState()
    # mu = torch.tensor([[0.]*12]*H)
    # sigma = torch.tensor([[0.2]*12]*H)
    mu = torch.tensor([[0.]*6]*H)
    sigma = torch.tensor([[0.2]*6]*H)
    count = 0

    # ________________LINE 0________________
    for _ in range(N):
        # ________________LINE 1________________
        # Sample G initial plans
        plans = []
        # Generate all the random actions for each plan
        for _ in range(G):
            plans.append(sampleNormDistr(quadruped, mu, sigma))

        # ________________LINE 2________________
        # Get initial states of the joints
        # Just using floating base state for now
        s0 = getState(quadruped)

        iters = 5
        jointIds = [2,3,4,6,7,8,10,11,12,14,15,16]   # all joints excluding foot, body, imu_joint

        # ________________LINE 3________________
        for i in range(iters):
            # ________________LINE 4b________________
            # get sequence states from sampled sequence of actions
            # initialize path set and fill it with G paths generated from the actions
            pathSet = []
            for g in range(G):
                state_sequence = getStateSeq(plans[g], H, quadruped, jointIds, currentID)
                pathSet.append((state_sequence, plans[g]))

            # ________________LINE 5________________
            # get cost of each path (sequence of states) generated above
            actionSetCosts = []
            for path in pathSet:
                cost = getPathCost(path[0], s0)
                actionSetCosts.append((path[1],cost))

            # ________________LINE 6________________
            # Sort the sequence of actions based on their cost
            sortedList = sorted(actionSetCosts, key = lambda x: x[1])
            # print("sortedList: \n", sortedList)
            
            # ________________LINE 7________________
            # Update normal distribution to fit top k action sequences

            # Take top K paths according to cost
            topKPath = [np.array(i[0]) for i in sortedList[0:k]]

            # Need if you make size of mu 6
            # top = []
            # for li in topKPath:
            #     li = [i[:6] for i in li]
            #     top.append(li)
            # topKPath = top

            a = np.average(np.array(topKPath), axis=0).tolist()
            b = np.std(np.array(topKPath), axis=0).tolist()
            mu = torch.tensor(a)
            sigma = torch.tensor(b)

            # ________________LINE 8________________
            # Replace bottom G-K action sequences with samples from
            bottomActions = []
            for _ in range(G-k):
                actions = sampleNormDistr(quadruped, mu, sigma)
                path = getStateSeq(actions, H, quadruped, jointIds, currentID)
                cost = getPathCost(path, s0)
                bottomActions.append((actions, cost))
            
            topActions = sortedList[0:k]
            actionSet = bottomActions + topActions
            actionSet = sorted(actionSet, key = lambda x: x[1])

        # ________________LINE 9________________
        # execute first action from the "best" model in the action sorted action sequences 
        # execute action
        bestAction = actionSet[0][0][0]
        applyAction(quadruped, jointIds, bestAction)
        finalActions.append(bestAction)

        # save state after
        currentID = p.saveState()


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


if __name__ == '__main__':
    main()


# [1,2,5.4,123,41]
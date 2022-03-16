from mimetypes import init
# from ssl import _PasswordType
import pybullet as p
import time
import os
import sys
import pybullet_data
import numpy as np
import torch


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


"""Get x,y,z coordinates of the hip joints."""
def getState(quadruped):
    FR_hip_joint = p.getLinkState(quadruped, 2)[0]
    FL_hip_joint = p.getLinkState(quadruped, 6)[0]
    RR_hip_joint = p.getLinkState(quadruped, 10)[0]
    RL_hip_joint = p.getLinkState(quadruped, 14)[0]
    # print('These are the FR, FL, RR, RL hip joints respectively: ', FR_hip_joint, FL_hip_joint, RR_hip_joint, RL_hip_joint)
    return [FR_hip_joint, FL_hip_joint, RR_hip_joint, RL_hip_joint]


"""Apply a random action to the all the links/joints of the hip."""
def applyAction(quadruped, jointIds, action):
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

    # calculate where the FR and FL hip's x coordinates are
    currX = (state[0][0] + state[1][0]) * 0.5
    prevX = (prevState[0][0] + prevState[1][0]) * 0.5

    # Calculating Tilt of Floating Body
    sum = 0 
    sum += abs(z1 - z2)
    sum += abs(z1 - z3)
    sum += abs(z1 - z4)
    sum += abs(z2 - z3)
    sum += abs(z2 - z4)
    sum += abs(z3 - z4)
    # get average
    tilt = sum/6
    
    # Calculating how high in air
    # Just want z-coordinate to stay same--not in air or crouching
    avgZ = (z1+z2+z3+z4)/4
    heightDiff = abs(avgZ - z_init)
    
    # Calculate if moving forward
    diffX = prevX - currX

    totalCost = 10*tilt + 10*heightDiff + 20*diffX
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


def sampleNormDistr(mu, sigma):
    action_sequence = torch.normal(mean=mu, std=sigma)
    return action_sequence


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

    # Initial Variables
    G = 30     # G is the number of paths generated (with the best 1 being picked)
    H = 60     # Number of states to predict per path (prediction horizon)
    N = 100     # How many iterations we're running the training for
    k = int(0.4 * G)    # Choosing the top k paths to create new distribution
    currentID = p.saveState()
    mu = torch.tensor([[0.]*12]*H)
    sigma = torch.tensor([[1.]*12]*H)
    count = 0

    # ________________LINE 0________________
    for _ in range(N):
        # ________________LINE 1________________
        # Sample G initial plans
        plans = []
        # Generate all the random actions for each plan
        for _ in range(G):
            plans.append(sampleNormDistr(mu, sigma))

        # print('________________THIS IS PLANS___________________\n', plans)
        # print('________________________________________________\n')

        # ________________LINE 2________________
        # Get initial states of the joints
        # Just using floating base state for now
        s0 = getState(quadruped)
        # print("Initial State \n", s0)

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
            a = np.average(np.array(topKPath), axis=0).tolist()
            b = np.std(np.array(topKPath), axis=0).tolist()
            mu = torch.tensor(a)
            sigma = torch.tensor(b)

            # ________________LINE 8________________
            # Replace bottom G-K action sequences with samples from
            bottomActions = []
            for _ in range(G-k):
                actions = sampleNormDistr(mu, sigma)
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
        bestAction = actionSet[0][0][1]
        applyAction(quadruped, jointIds, bestAction)
        bestAction = actionSet[0][0][2]
        applyAction(quadruped, jointIds, bestAction)
                
        # save state after
        currentID = p.saveState()


        print("Iteration: ", count)
        print("This is the ID: ", currentID)
        count += 1
        # exit()



        
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
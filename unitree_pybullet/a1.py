import argparse
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
    
    return urdfFlags, quadruped

"""For Milestone 1."""
def getEnvInfo(quadruped):
    initPosition, initOrientation = p.getBasePositionAndOrientation(quadruped)
    envInfo = {
        'position': initPosition, 
        'orientation': initOrientation
    }
    return envInfo

"""For Milestone 1."""
def getRobotInfo(quadruped):
    numJoints = p.getNumJoints(quadruped)
    jointInfo = {}
    jointStates = {}
    for j in range(numJoints):
        jointInfo[j] = p.getJointInfo(quadruped, j)
        jointStates[j] = p.getJointState(quadruped, j)    # jointPosition, jointVelocity, jointReactionForces, appliedJointMotorTorque

    return jointInfo, jointStates

"""Gets the upper and lower positional limits of each joint."""
def getJointsRange(uid, jointIds):
    jointsRange = []
    for a in jointIds:
        jointInfo = p.getJointInfo(uid, a)
        jointsRange.append((jointInfo[8], jointInfo[9]))
    return jointsRange

"""Gets the maxForce that should be applied to each joint."""
def getJointsMaxForce(uid, jointIds):
    jointsMaxForces = []
    for j in jointIds:
        jointInfo = p.getJointInfo(uid, j)
        jointsMaxForces.append(jointInfo[10])
    return jointsMaxForces

"""Apply a random action to the all the links/joints of the hip."""
def applyAction(quadruped, jointIds, action):
    # Makes the action read from the toggle bar for forces
    # Feel like this should be proportional to each joint and not the same for every joint. 
    # fs = [p.readUserDebugParameter(maxForceId)] * 12      
    # p.setJointMotorControlArray(quadruped, jointIds, p.POSITION_CONTROL, action, forces=fs)

    p.setJointMotorControlArray(quadruped, jointIds, p.POSITION_CONTROL, action)
    for _ in range(10):
        p.stepSimulation()

"""Calculate other methods to tweak the cost function."""
# def getOtherCosts(jointIds, actionSeq, s0, H, hip_s0):
#     z_init = 0.480031    # Z-coord of hip joint at initial state
#     # z_prev = (prevState[0][2] + prevState[1][2] + prevState[3][2] + prevState[4][2])/4 
#     z1 = hip_s0[0][2]
#     z2 = hip_s0[1][2]
#     z3 = hip_s0[2][2]
#     z4 = hip_s0[3][2]

#     ##### Calculating Tilt of Floating Body
#     sums = 0
#     sums += abs(z1 - z2)
#     sums += abs(z1 - z3)
#     sums += abs(z1 - z4)
#     sums += abs(z2 - z3)
#     sums += abs(z2 - z4)
#     sums += abs(z3 - z4)
#     body_tilt = sums/6   # get average
#     #####

#     # Calculating how high in air
#     # Just want z-coordinate to stay same--not in air or crouching
#     avgZ = (z1+z2+z3+z4)/4
#     heightDiff = abs(avgZ - z_init)
    
#     ##### Calculate FR and FL hip's x,y coordinates
#     # calculate where the FR and FL hip's x coordinates are
#     currX = (state[0][0] + state[1][0]) * 0.5
#     prevX = (hip_s0[0][0] + hip_s0[1][0]) * 0.5
#     diffX = prevX - currX   # Calculate if moving forward

#     # calculate where the FR and FL hip's y coordinates are
#     diffY = (abs(state[0][1] + state[1][1]) + abs(state[2][1] + state[3][1])) * 0.5
#     # diffY = currY    # Calculate if straight
#     #####

#     ##### Calculate cost from target_x
#     # Want it to move 5 units in x direction
#     target_x = []
#     for tup in s0:
#         t = (tup[0] + 5, tup[1], tup[2])
#         target_x.append(t)

#     dist_cost = 0
#     for n in range(4):
#         dist_cost += dist(list(state[n]), list(s0[n]))

#     # print(s0)
#     # print(state)
#     # print(dist_cost)

#     # totalCost = 10*body_tilt + 10*heightDiff + 20*diffX
#     # totalCost = 100*body_tilt + 100*heightDiff + 100*diffX + 30*diffY + foot_diff*100
#     # totalCost = 100*body_tilt + 1000*heightDiff + 100*diffX + 100*diffY + foot_diff*100
#     totalCost = 30*body_tilt + 10*heightDiff + 40*diffX + 30*diffY
#     return totalCost


"""Applies an action to each joint, then returns the position of the floating base."""
def getState(quadruped, jointIds, action):    # Changed
    applyAction(quadruped, jointIds, action)
    floating_base_pos = p.getLinkState(quadruped, 0)[0] # a1's floating base center of mass position
    return floating_base_pos

"""Calculates the total cost of each path/plan."""
"""NOTE: only calculating +0.5 x-dist as cost."""
def getPathCost(quadruped, jointIds, actionSeq, s0, H, hip_s0):
    resolution = 0.5
    hFutureStates = []
    x, y, z = s0
    for _ in range(H):
        x += resolution
        hFutureStates.append((x, y, z))
    actionSeq = actionSeq.reshape(H, -1)
    cost = 0
    state = getState(quadruped, jointIds, actionSeq[0])  # Only doing first action bc total errors make it skewed
    target = hFutureStates[0]
    cost += dist(state, target)

    # Calculate costs for all the actions of the plan
    # for h in range(H):
    #     state = getState(quadruped, jointIds, actionSeq[h])
    #     target = hFutureStates[h]
    #     cost += dist(state, target)
    #     cost += getOtherCosts(jointIds, actionSeq, s0, H, hip_s0)
    # print("COST: ", cost)
    return cost

"""Samples random points from normal distribution."""
def sampleNormDistr(jointsRange, mu, sigma, G, H):
    mins, maxes = np.array(jointsRange).T
    actionSeqSet = [] # action sequence set that contains g sequences
    for g in range(G):
        samp = np.random.multivariate_normal(mu.numpy(), sigma.numpy()).reshape(H, len(mins))
        samp = np.clip(samp, mins, maxes)
        actionSeqSet.append(torch.tensor(samp.reshape(-1)))
    return torch.stack(actionSeqSet)


def main():
    urdfFlags, quadruped = loadA1()
    envInfo = getEnvInfo(quadruped)
    jointInfo, jointStates = getRobotInfo(quadruped)
    finalActions = []
    jointIds = [2,3,4,6,7,8,10,11,12,14,15,16]   # all joints excluding foot, body, imu_joint
    jointsRange = getJointsRange(quadruped, jointIds)

    ####### Milestone 2 ######

    # Initial Variables
    N = 100                 # How many iterations we're running the training for
    T = 5                   # Number of training iteration
    G = 10                  # G is the number of paths generated (with the best 1 being picked)
    H = 5                   # Number of states to predict per path (prediction horizon)
    K = int(0.4 * G)        # Choosing the top k paths to create new distribution

    # Initial mean and std. dev
    mu = torch.zeros(H, len(jointIds)).flatten() # Changed
    sigma = np.pi * torch.eye(len(mu))
    
    count = 0   # debugging

    # ________________LINE 0________________
    for _ in range(N):
        currentID = p.saveState() # Save the state before simulations. # Changed

        # ________________LINE 1file:///homes/chen4066/Downloads/sanchez-gonzalez18a.pdf________________
        # Sample G initial plans and generate all the random actions for each plan
        plans = sampleNormDistr(jointsRange, mu, sigma, G, H)

        # ________________LINE 2________________
        # Use floating base center of mass initial state position for now to compare
        s0 = p.getLinkState(quadruped, 0)[0]
        hip_s0 = []
        hip_s0.append(p.getLinkState(quadruped, 2)[0])
        hip_s0.append(p.getLinkState(quadruped, 6)[0])
        hip_s0.append(p.getLinkState(quadruped, 10)[0])
        hip_s0.append(p.getLinkState(quadruped, 14)[0])


        # ________________LINE 3________________
        for _ in range(T):
            # print("Start: ", p.getLinkState(quadruped, 0)[0]) # a1's floating base center of mass position

            # ________________LINE 4b________________
            # Get sequence states from sampled sequence of actions
            # ________________LINE 5________________
            # Get cost of each path (sequence of states)
            actionSetCosts = []
            for plan in plans:
                # Restore back to original state to run the plan again
                p.restoreState(currentID)
                # getPathCost - applies action, gets that state, returns path cost
                cost = getPathCost(quadruped, jointIds, plan, s0, H, hip_s0)
                actionSetCosts.append((plan, cost))

            # ________________LINE 6________________
            # Sort the sequence of actions based on their cost
            sortedList = sorted(actionSetCosts, key = lambda x: x[1])
            
            # ________________LINE 7________________
            # Update normal distribution to fit top k action sequences

            # Each entry is all of one action, i.e. a_0
            sep_actions = []
            for j in range(K):
                sep_actions.append(sortedList[j][0])
            sep_actions = torch.stack(sep_actions)

            # Get the mean and std dev. of each action a_i
            mu = torch.mean(sep_actions, dim=0)
            sigma = torch.cov(sep_actions.T)
            sigma += .02 * torch.eye(len(mu))   # add a small amount of noise to the diagonal to replan to next target

            # ________________LINE 8________________
            # Replace bottom G-K action sequences with samples from refined distribution above
            refined_actions = sampleNormDistr(jointsRange, mu, sigma, G-K, H)
            actionSeqSet = torch.cat((sep_actions, refined_actions))

        # ________________LINE 9________________
        # Execute first action from the "best" model in the action sorted action sequences 
        bestAction = actionSeqSet[0][:len(jointIds)]
    
        p.restoreState(currentID)   # Before applying action, restore state to previous
        a1Pos = p.getLinkState(quadruped, 0)[0]
        p.resetDebugVisualizerCamera( cameraDistance=2, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=a1Pos)
        applyAction(quadruped, jointIds, bestAction)
        finalActions.append(bestAction.tolist())    # Keep track of all actions

        # print("End: ", p.getLinkState(quadruped, 0)[0])
        print("Iteration: ", count)
        count += 1

    # Write to actions
    filename = "final_actions.csv"
    # Writing to csv file
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
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

    jointIds = [2,3,4,6,7,8,10,11,12,14,15,16]   # all joints excluding foot, body, imu_joint

    for env_step in range(len(finalActions)):
        a1Pos = p.getLinkState(quadruped, 0)[0]
        p.resetDebugVisualizerCamera( cameraDistance=2, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=a1Pos)
        applyAction(quadruped, jointIds, finalActions[env_step])
        # time.sleep(1./125.)
        time.sleep(1./25.)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CS 593-ROB - Project Milestone 2')
    parser.add_argument('-p', '--play', action='store_true', help='Set true to playback the recorded best actions.')

    args = parser.parse_args()
    
    if args.play:
        playback()
    else: 
        main()

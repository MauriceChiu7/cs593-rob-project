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

# L2 Norm between 2 Vectors
def dist(p1, p2):
    return magnitude(diff(p1, p2))

def normalize(vector):
    normalized = vector/np.linalg.norm(vector)
    return normalized

# Returns the joint force range corresponding to each inputted jointID
def getJointForceRange(uid, jointIds):
    forces = getJointsMaxForce(uid, jointIds)
    jointsForceRange = []
    for f in forces:
        mins = -f
        maxs = f
        jointsForceRange.append((mins, maxs))
    jointsForceRange = [(-2, 2)] * 12   # put a limit to how much each foot moves
    return jointsForceRange

# creates initial distribution with mu of 0's and covariance that is the identity matrix
def initialDist(uid, jointIds, G, H):
    mu = torch.zeros(H, len(jointIds)).flatten()
    sigma = np.pi * torch.eye(len(mu))
    return (mu, sigma)

# def refineDist(mu, sigma):
#     # print(mu)
#     mu = mu[12:]
#     zeroes = torch.zeros(12)
#     mu = torch.cat([mu,zeroes])
#     # print(mu)
#     temp_sig = np.pi * torch.eye(len(mu))
#     for i in range(len(temp_sig)-1):
#         for j in range(len(temp_sig[0])-1):
#             temp_sig[i][j] = sigma[i+1][j+1]

#     return mu,temp_sig

def loadA1(train):
    # only render when doing playback, not when training!
    if train:
        p.connect(p.DIRECT)
    else:
        p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), 0, 0, 0)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1./100)
    urdfFlags = p.URDF_USE_INERTIA_FROM_FILE
    quadruped = p.loadURDF("data/a1/urdf/a1.urdf",[0,0,0.48], p.getQuaternionFromEuler([0,0,0]), flags=urdfFlags)
    #enable collision between lower legs
    lower_legs = [2,5,8,11]
    for l0 in lower_legs:
        for l1 in lower_legs:
            if (l1>l0):
                enableCollision = 1
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

"""Gets the maxForce that should be applied to each joint."""
def getJointsMaxForce(uid, jointIds):
    jointsMaxForces = []
    for j in jointIds:
        jointInfo = p.getJointInfo(uid, j)
        jointsMaxForces.append(jointInfo[10])
    return jointsMaxForces

"""Apply a random action to the all the links/joints of the hip."""
def applyAction(quadruped, jointIds, action):
    action = torch.mul(action, 50)
    p.setJointMotorControlArray(quadruped, jointIds, p.POSITION_CONTROL, targetPositions=action)
    for _ in range(10):
        p.stepSimulation()

def applyActionPB(quadruped, jointIds, action):
    action = torch.mul(torch.Tensor(action), 50)
    p.setJointMotorControlArray(quadruped, jointIds, p.POSITION_CONTROL, targetPositions=action)
    for _ in range(10):
        p.stepSimulation()

"""Applies an action to each joint, then returns the position of the floating base."""
def getState(quadruped, jointIds, action):
    applyAction(quadruped, jointIds, action)
    floating_base_pos = p.getLinkState(quadruped, 0)[0] # a1's floating base center of mass position
    return floating_base_pos

def getIKsolver(quadruped, footIds, actionSeq):
    actionSeq = actionSeq.tolist()

    ind = 0
    posits = []
    for ft in footIds:
        ft_state = p.getLinkState(quadruped, ft)[0]
        posits.append([ft_state[0]+actionSeq[ind], ft_state[1] + actionSeq[ind+1], ft_state[2] + actionSeq[ind+2]])
        ind += 2

    # u = p.calculateInverseKinematics(quadruped, 5, posits[0])
    fs = p.calculateInverseKinematics2(quadruped, footIds, posits)
    # print("forces: ", fs)
    # print("u: ", u)
    # exit()

    # print(torch.Tensor(fs))
    # exit()
    return torch.Tensor(fs)


"""Calculates the total cost of each path/plan."""
def getPathCost(quadruped, footIds, jointIds, actionSeq, H, Goal):
    weights = [30,5,30]
    z_init = 0.480031
    # Reshape action sequence to array of arrays (originally just a single array)
    actionSeq = actionSeq.reshape(H, -1)

    # Initialize cost
    cost = 0
    # Loop through each action of the plan and add cost
    for h in range(H):
        currAction = getIKsolver(quadruped, footIds, actionSeq[h])
        # currAction = actionSeq[h]
        state = getState(quadruped, jointIds, currAction)
        distCost = weights[0] * dist(state, Goal) # distance from goal
        actionCost = weights[1] * magnitude(actionSeq[h]) # gets the magnitude of actions (shouldn't apply huge actions)
        zCost = weights[2] * abs(state[2]-z_init)
        cost += distCost   
        cost += actionCost
        cost += zCost
        # print(f"dist cost: {distCost}, action cost: {actionCost}")
    return cost

"""Samples random points from normal distribution."""
def sampleNormDistr(jointsRange, mu, sigma, G, H):
    mins, maxes = np.array(jointsRange).T
    actionSeqSet = [] # action sequence set that contains g sequences
    for g in range(G):
        samp = np.random.multivariate_normal(mu.numpy(), sigma.numpy()).reshape(H, len(mins))
        # samp = np.clip(samp, mins*1000, maxes*1000)
        actionSeqSet.append(torch.tensor(samp.reshape(-1)))
    return torch.stack(actionSeqSet)


def train():
    train = True
    urdfFlags, quadruped = loadA1(train)
    finalActions = []
    jointIds = [2,3,4,6,7,8,10,11,12,14,15,16]   # all joints excluding foot, body, imu_joint
    footIds = [5, 9, 13, 17]   # only feet
    jointsRange = getJointForceRange(quadruped, jointIds)

    ####### Milestone 2 ######

    # Initial Variables
    N = args.N              # How many iterations we're running the training for
    T = args.T              # Number of training iteration
    G = args.G              # G is the number of paths generated (with the best 1 being picked)
    H = 10                  # Number of states to predict per path (prediction horizon)
    K = int(0.3 * G)        # Choosing the top k paths to create new distribution
    Goal = (100, 0, p.getLinkState(quadruped, 2)[0][2])
    print("\nGOAL: ", Goal)

    mu, sigma = initialDist(quadruped, jointIds, G, H)
    count = 0   # debugging

    # ________________LINE 0________________
    for _ in range(N):
        currentID = p.saveState() # Save the state before simulations. # Changed

        # ________________LINE 1________________
        # Sample G initial plans and generate all the random actions for each plan
        plans = sampleNormDistr(jointsRange, mu, sigma, G, H)

        # ________________LINE 2________________
        # Use floating base center of mass initial state position for now to compare

        # ________________LINE 3________________
        for _ in range(T):
            # ________________LINE 4b and 5________________
            # Get sequence states from sampled sequence of actions
            # Get cost of each path (sequence of states)
            actionSetCosts = []
            for plan in plans:
                # Restore back to original state to run the plan again
                p.restoreState(currentID)
                # getPathCost - applies action, gets that state, returns path cost
                cost = getPathCost(quadruped, footIds, jointIds, plan, H, Goal)
                actionSetCosts.append((plan, cost))

            # ________________LINE 6________________
            # Sort the sequence of actions based on their cost
            sortedList = sorted(actionSetCosts, key = lambda x: x[1])
            # print(sortedList)
            
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
            sigma += .03 * torch.eye(len(mu))   # add a small amount of noise to the diagonal to replan to next target

            # ________________LINE 8________________
            # Replace bottom G-K action sequences with samples from refined distribution above
            refined_actions = sampleNormDistr(jointsRange, mu, sigma, G-K, H)
            actionSeqSet = torch.cat((sep_actions, refined_actions))
            plans = actionSeqSet

        # ________________LINE 9________________
        # Execute first action from tvfhe "best" model in the action sorted action sequences 
        bestAction = actionSeqSet[0][:len(jointIds)]    
        p.restoreState(currentID)   # Before applying action, restore state to previous
        applyAction(quadruped, jointIds, bestAction)
        finalActions.append(bestAction.tolist())    # Keep track of all actions

        # mu,sigma = refineDist(mu,sigma)
        # mu,sigma = initialDist(quadruped, jointIds, G, H)

        print("Env_step: ", count)
        # print("THIS IS MU: ", mu)
        # print("THIS IS SIGMA: ", sigma)
        count += 1

    # Write to actions
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)
    
    filename = args.dir + "results_G" + str(args.G) + "_T" + str(args.T) + "_N" + str(args.N)
    # Writing to csv file
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, lineterminator = '\n')
        csvwriter.writerows(finalActions)
    print("Done Training")


def playback():
    if args.f != 'NA':
        filename = args.dir + args.f
    else:
        filename = args.dir + "results_G" + str(args.G) + "_T" + str(args.T) + "_N" + str(args.N)
    f = open(filename)
    csvreader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
    finalActions = []
    for row in csvreader:
        finalActions.append(row)
    # print(finalActions)
    f.close()

    flag, quadruped = loadA1(False)
    for _ in range(100):
        p.stepSimulation()

    p.setRealTimeSimulation(0)
    p.setTimeStep(1./100)

    jointIds = [2,3,4,6,7,8,10,11,12,14,15,16]   # all joints excluding foot, body, imu_joint

    for env_step in range(len(finalActions)):
        # a1Pos = p.getLinkState(quadruped, 0)[0]
        # p.resetDebugVisualizerCamera( cameraDistance=2, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=a1Pos)
        applyActionPB(quadruped, jointIds, finalActions[env_step])
        time.sleep(1./3.)
        # time.sleep(.5)

def test():
    flag, quadruped = loadA1(False)
    jointIds = [2,3,4,6,7,8,10,11,12,14,15,16]   # all joints excluding foot, body, imu_joint
    # stuff = getJointForceRange(quadruped, jointIds)
    # print("THESE ARE THE FORCE RANGES: ", stuff)
    for _ in range(100):
        p.stepSimulation()

    p.setRealTimeSimulation(0)
    p.setTimeStep(1./100)




    for _ in range(50):
        applyActionPB(quadruped, jointIds, [3,0,0,0,0,0,0,0,0,0,0,0])
    while(1):
        applyActionPB(quadruped, jointIds, [0,0,-.1,0,0,0,0,0,0,0,0,0])
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CS 593-ROB - Project Milestone 2')
    parser.add_argument('-dir', type=str, default='./results/',help='folder to read data from')
    parser.add_argument('-G', type=int, default=150,help='Num plans')
    parser.add_argument('-T', type=int, default=5,help='Training Iterations for 1 action')
    parser.add_argument('-N', type=int, default=150,help='Num Env Steps')
    parser.add_argument('-f', type=str, default='NA', help='Specific filename for playback')
    parser.add_argument('-p', '--play', action='store_true', help='Set true to playback the recorded best actions.')
    parser.add_argument('-t', '--test', action='store_true', help='Set true to test stuff.')
    args = parser.parse_args()
       
    if args.play:
        playback()
    elif args.test:
        test()
    else: 
        train()

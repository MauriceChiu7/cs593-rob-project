import pybullet as p
import pybullet_data
import os
import torch
import numpy as np
import math
import pickle
import time
import random
import sys

ACTIVE_JOINTS = [1,2,3,4,5,6,8,9] # All of the movable joints of the UR5
END_EFFECTOR_INDEX = 7 # The end effector link index of the UR5.
ELBOW_INDEX = 3 # The elbow link index of the UR5.
DISCRETIZED_STEP = 0.05 # The step size for discretizing a trajectory.

def diff(v1, v2):
    """
    Description: Calculates the difference between two n-vectors
    
    Input:
    :v1 - {Tensor} a n-vector
    :v2 - {Tensor} a n-vector
    
    Returns:
    :torch.sub(v1, v2) - {Tensor} the difference between the two n-vectors
    """
    return torch.sub(v1, v2)

def magnitude(v):
    """
    Description: Calculates the magnitude of a vector
    
    Input:
    :v - {Tensor} a n-vector
    
    Returns:
    :torch.sqrt(torch.sum(torch.pow(v, 2))) - {Tensor} the magnitude of the n-vector
    """
    return torch.sqrt(torch.sum(torch.pow(v, 2)))

def dist(p1, p2):
    """
    Description: Calculates distance between two vectors
    
    Input:     
    :p1 - {Tensor} a n-vector
    :p2 - {Tensor} a n-vector
    
    Returns:
    :magnitude(diff(p1, p2)) - {Tensor} the distance between the two n-vectors
    """
    return magnitude(diff(p1, p2))

def loadEnv():
    """
    Description: Loads pybullet environment with a horizontal plane and earth like gravity.
    """
    p.connect(p.DIRECT)
    # p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0.1])
    p.setGravity(0, 0, -9.8)
    # p.setTimeStep(1./50.)
    p.setTimeStep(1./20/3)

def loadUR5():
    """
    Description: Loads the UR5 robot into the environment
    
    Returns:
    :uid - {Int} body unique id of the robot
    """
    p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=(0,0,0))
    path = f"{os.getcwd()}/../ur5pybullet"
    print(path)

    os.chdir(path) # Needed to change directory to load the UR5.
    uid = p.loadURDF(os.path.join(os.getcwd(), "./urdf/real_arm.urdf"), [0.0,0.0,0.0], p.getQuaternionFromEuler([0,0,0]), flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION)
    path = f"{os.getcwd()}/../ur5"
    os.chdir(path) # Back to parent directory.
    # Enable collision for all link pairs.
    for l0 in range(p.getNumJoints(uid)):
        for l1 in range(p.getNumJoints(uid)):
            if (not l1>l0):
                enableCollision = 1
                # print("collision for pair",l0,l1, p.getJointInfo(uid,l0)[12],p.getJointInfo(uid,l1)[12], "enabled=",enableCollision)
                p.setCollisionFilterPair(uid, uid, l1, l0, enableCollision)
    return uid

def getConfig(uid, jointIds):
    """
    Description: Gets the position (in radians) of the joint(s)
    
    Input:
    :uid - {Int} body unique id
    :jointIds - {List} a list of joint ids which you want to get the position of
    
    Returns:
    :torch.Tensor(config) - {Tensor} a list of joint positions in radians
    """
    config = []
    for id in jointIds:
        # print(p.getJointState(uid, id)[0])
        config.append(p.getJointState(uid, id)[0])
    # EEPos = getState(uid)[0].tolist()
    # config.append(EEPos[0])
    # config.append(EEPos[1])
    # config.append(EEPos[2])
    return torch.Tensor(config)

def getLimitPos(jointIds, uid):
    """
    Description: Gets the upper and lower positional limits of each joint

    Input:
    :jointIds - {List} a List of joint ids
    :uid - {Int} body unique id of the robot

    Returns:    
    :mins - {List} a list of lower ranges
    :maxes - {List} a list of upper ranges
    """
    mins = []
    maxes = []
    for id in jointIds:
        info = p.getJointInfo(uid, id)
        mins.append(info[8])
        maxes.append(info[9])
    return mins, maxes

def getJointsRange(uid, jointIds):
    """
    Description: Gets the upper and lower positional limits of each joint

    Input:
    :uid - {Int} body unique id of the robot
    :jointIds - {List} a list of joint ids

    Returns:    
    :jointsRange - {List} a list of tuples that contains the lower and upper ranges of joints
    """
    jointsRange = []
    for a in jointIds:
        jointInfo = p.getJointInfo(uid, a)
        jointsRange.append((jointInfo[8], jointInfo[9]))
    return jointsRange

def randomInit(uid):
    """
    Description: Sample random positions for every active joints of the robot and make the robot get into that pose

    Input:
    :uid - {Int} body unique id of the robot

    Returns:
    :initState - {List} a list of Tensors that is the position of every active joint.
    :initCoords - {Tensor} the coordinates of the end-effector
    """
    # Start ur5 with random positions
    jointsRange = getJointsRange(uid, ACTIVE_JOINTS)
    random_positions = []
    for r in jointsRange:
        rand = np.random.uniform(r[0], r[1])
        random_positions.append(rand)
    random_positions = torch.Tensor(random_positions)
    applyAction(uid, random_positions)
    initState = getConfig(uid, ACTIVE_JOINTS)
    initCoords = torch.Tensor(p.getLinkState(uid, END_EFFECTOR_INDEX, 1)[0])
    p.addUserDebugLine([0,0,0.1], initCoords, [1,0,0])
    # time.sleep(10)
    return initState, initCoords

def randomGoal():
    """
    Description: Generates a random goal point for the end-effector

    Returns:
    :goalCoords - {Tensor} the goal coordinates for the end-effector
    """
    # Generate random goal state for UR5
    x = np.random.uniform(-0.7, 0.7)
    y = np.random.uniform(-0.7, 0.7)
    z = np.random.uniform(0.15, 0.7)
    goalCoords = torch.Tensor([x, y, z])
    p.addUserDebugLine([0,0,0.1], goalCoords, [0,0,1])
    return goalCoords

def makeTrajectory(initCoords, goalCoords):
    """
    Description: Make two given points in 3-D space into a discretized trajectory

    Input:
    :initCoords - {Tensor} the trajectory's starting point
    :goalCoords - {Tensor} the trajectory's ending point

    Returns: 
    :torch.stack(traj) - {Tensor} a Tensor sequence that contains all the point/coordinates of the trajectory
    """
    distTotal = dist(goalCoords, initCoords)
    diffBtwin = diff(goalCoords, initCoords)
    incrementTotal = torch.div(distTotal, DISCRETIZED_STEP)
    numSegments = int(math.floor(incrementTotal))+1
    stepVector = diffBtwin / numSegments

    # print("initCoords:\t", initCoords)
    # print("goalCoords:\t", goalCoords)
    # print("distTotal:\t", distTotal)
    # print("diffBtwin:\t", diffBtwin)
    # print("incrementTotal:\t", incrementTotal)
    print()
    print("numSegments:\t", numSegments)
    
    traj = []
    for i in range(numSegments+1):
        # print(torch.mul(stepVector, i))
        traj.append(torch.add(initCoords, torch.mul(stepVector, i)))
    
    for pt in range(len(traj)-1):
        p.addUserDebugLine([0,0,0.1],traj[pt],traj[pt+1])
    
    return torch.stack(traj)

def getState(uid):
    """
    Description: returns the coordinates of the end-effector and the elbow joint

    Input:
    :uid - {Int} body unique id of the robot

    Returns:
    :state - {Tensor} a tensor sequence containing the end-effector and elbow joint coordinates
    """
    eePos = p.getLinkState(uid, END_EFFECTOR_INDEX)[0]
    elbowPos = p.getLinkState(uid, ELBOW_INDEX)[0]
    state = torch.Tensor([eePos, elbowPos])
    return state

def getJointPos(uid):
    """
    Description: returns a list of coordinates of every active joints of the robot

    Input:
    :uid - {Int} body unique id of the robot

    Returns:
    :jointPos - {List} a list of joint coordinates
    """
    jointStates = p.getLinkStates(uid, ACTIVE_JOINTS)
    jointPos = []
    for j in jointStates:
        x, y, z = j[0]
        jointPos.append([x, y, z])
    jointPos[1] = torch.sub(torch.Tensor(jointPos[1]),torch.mul(diff(torch.Tensor(jointPos[1]), torch.Tensor(jointPos[4])), 0.3)).tolist()
    return jointPos

def getReward(action, jointIds, uid, target, distToGoal):
    """
    Description: calculates the cost/reward of an action

    Input:
    :action - {Tensor} a tensor sequence containg the position of all active joints
    :jointIds - {List} a list of joint ids
    :uid - {Int} body unique id of the robot
    :target - {Tensor} the next way point for the end-effector
    :distToGoal - {Float} the distance from the current end-effector coordinates to the goal coordinates

    Returns:
    :reward - {Tensor} the calculated cost/reward of the given action
    """
    state = getState(uid)
    applyAction(uid, action)

    next_state = getState(uid)
    distCost = dist(next_state[0], target)
    elbowCost = dist(next_state[1], state[1])
    groundColliCost = 0
    jointPositions = getJointPos(uid)
    jointZs = []
    for pos in jointPositions:
        jointZs.append(pos[2])
        if pos[2] < 0.15:
            groundColliCost += 1

    weight = torch.Tensor([10, 1, 2])
    rawCost = torch.Tensor([distCost, elbowCost, groundColliCost])
    reward = (weight * rawCost).sum().numpy()
    if distCost > distToGoal: 
        reward += 10

    return reward

def getEpsReward(episode, jointIds, uid, Horizon, goalCoords, distToGoal):
    """
    Description: calculates the cost/reward of an entire episode

    Input:
    :episode - {Tensor} a tensor sequence of joint positions sampled from the multivariate normal distribution
    :jointIds - {List} a list of joint ids
    :uid - {Int} body unique id of the robot
    :Horizon - {Int} the horizon length for the MPC-CEM
    :goalCoord - {Tensor} the coordinates of the goal for the end-effector
    :distToGoal - {Tensor} the remaining distance from the end-effector's position to goal

    Returns:
    :reward - {Tensor} the total cost/reward of the entire episode
    """
    numJoints = len(jointIds)
    reward = 0
    for h in range(Horizon):
        start = h * numJoints
        end = start + numJoints
        action = episode[start:end]
        action = torch.Tensor(action)
        # reward += getReward(action, jointIds, uid, futureStates[h])
        reward += getReward(action, jointIds, uid, goalCoords, distToGoal)
    return reward

def applyAction(uid, action):
    """
    Description: make the robot get into pose with the action given

    Input:
    :uid - {Int} body unique id of the robot
    :action - {Tensor} a tensor sequence that contains the position of joints
    """
    p.setJointMotorControlArray(uid, ACTIVE_JOINTS, p.POSITION_CONTROL, action)
    maxSimSteps = 150
    for s in range(maxSimSteps):
        p.stepSimulation()
        currConfig = getConfig(uid, ACTIVE_JOINTS)
        action = torch.Tensor(action)
        currConfig = torch.Tensor(currConfig)
        error = torch.sub(action, currConfig)
        done = True
        for e in error:
            if abs(e) > 0.02:
                done = False
        if done:
            # print(f"reached position: \n{action}, \nwith target:\n{currConfig}, \nand error: \n{error} \nin step {s}")
            break

def main():
    """
    Description: the main MPC-CEM algorithm

    Writes: 
    :ur5sample_{pathNum}.pkl - the state-action-state pair for training the neural network
    :debug_{pathNum}.pkl - records of the initial state, initial end-effector coordinates, and the goal coordinates
    :finalEePos_{pathNum}.pkl - The final end-effector positions
    """
    initialTime = time.time() # Starting time

    # Random seed every run so the neural net is trained correctly
    torch_seed = np.random.randint(low=0, high=1000)
    np_seed = np.random.randint(low=0, high=1000)
    py_seed = np.random.randint(low=0, high=1000)
    torch.manual_seed(torch_seed)
    np.random.seed(np_seed)
    random.seed(py_seed)

    # File paths
    trainingFolder = "./trainingData/"
    errorFolder = "./error/"
    if not os.path.exists(trainingFolder):
        os.makedirs(trainingFolder)
    if not os.path.exists(errorFolder):
        os.makedirs(errorFolder)
    
    loadEnv() # Setting up pybullet and loading in the environment
    uid = loadUR5() # Loading the UR5 into the environment
    
    # jointsRange = getJointsRange(uid, ACTIVE_JOINTS)

    goalCoords = randomGoal() # Generate a random goal coordinates
    initState, initCoords = randomInit(uid) # Initialize the UR5 into a random pose

    # Saving these information for playback
    debug = {
        'goalCoords': goalCoords,
        'initState': initState,
        'initCoords': initCoords
    }

    # Calculate the distance from initial coordinates to goal coordinates
    distToGoal = dist(torch.Tensor(initCoords), torch.Tensor(goalCoords))

    # traj = makeTrajectory(initCoords, goalCoords)
    print("\n\ninitCoords:\t", initCoords)
    print("initState:\t", initState)
    print("goalCoords:\t", goalCoords)
    print("distToGoal:\t", distToGoal)
    # print("traj:\n", traj)
    
    # Constants:
    MAX_ITERATIONS = 3 # Program will quit after failing to reach goal for this number of iterations
    # Iterations = len(traj) # N - envSteps
    Iterations = MAX_ITERATIONS # N - envSteps
    Epochs = 20 # T - trainSteps
    Episodes = 1200 # G - plans
    Horizon = 1 # H - horizonLength
    TopKEps = int(0.15*Episodes) # how many episodes to use to calculate the new mean and covariance for the next epoch

    print(f"Iterations: {Iterations}, Epochs: {Epochs}, Episodes: {Episodes}, Horizon: {Horizon}, TopKEps: {TopKEps}")

    # Getting the joints' lower and upper limits
    jointMins,jointMaxes = getLimitPos(ACTIVE_JOINTS, uid)
    jointMins = jointMins*Horizon
    jointMaxes = jointMaxes*Horizon
    jointMins = torch.Tensor(jointMins)
    jointMaxes = torch.Tensor(jointMaxes)

    # List of final state-action-state pairs
    saveRun = []
    saveAction = []

    # Final end-effector positions
    finalEePos = []

    # Time per iteration taken
    iterationTimes = []

    # The loop for stepping through each environment steps
    for envStep in range(Iterations):
        startTime = time.time()
        print(f"Running Iteration {envStep} ...")
        
        # Calculate distance from start to goal
        eePos = getState(uid)[0]
        distError = dist(eePos, goalCoords)
        print("prevDistToGoal:\t", distError)

        # Initialize mean and covariance
        mu = torch.Tensor([0]*(len(ACTIVE_JOINTS) * Horizon))
        cov = torch.eye(len(mu)) * ((np.pi/2) ** 2)
        
        # Saving and restoring state to avoid a syncing issue with pybullet
        startState = p.saveState()
        p.restoreState(startState)
        
        # Saving the state before applying test actions for the MPC-CEM process
        stateId = p.saveState()

        # Get next h future states (The horizon length) 
        # Needed if there is a trajectory to follow
        # futureStates = []
        # for h in range(Horizon):
        #     if envStep + h > len(traj) - 1:
        #         futureStates.append(traj[-1])
        #     else:
        #         futureStates.append(traj[envStep + h])
        # futureStates = torch.stack(futureStates)
        # print("futureStates:\n", futureStates)

        epsMem = [] # List to store all episodes and their associated costs
        
        # The loop for improving the distribution
        for e in range(Epochs):
            print(f"Epoch {e}")
            # Initialize the distribution which we sample our actions from
            distr = torch.distributions.MultivariateNormal(mu, cov)
            
            # The loop for generating all the required episodes
            for eps in range (Episodes):
                p.restoreState(stateId) # Restore to saved state before testing our actions
                episode = distr.sample() # Sample an episode of actions
                
                # Make sure samples don't exceed the joints' limit
                episode = torch.clamp(episode, jointMins, jointMaxes).tolist()
                
                # Calculates the cost of each episode
                cost = getEpsReward(episode, ACTIVE_JOINTS, uid, Horizon, goalCoords, distToGoal)
                # cost = getEpsReward(episode, ACTIVE_JOINTS, uid, Horizon, futureStates)
                
                epsMem.append((episode,cost)) # Save the episodes and their associated costs
            
            # Restore to saved state before applying the best action
            p.restoreState(stateId)

            # Sort episodes by cost, ascending
            epsMem = sorted(epsMem, key = lambda x: x[1])

            # Keep the top K episodes
            epsMem = epsMem[0:TopKEps]

            # print("epsMem: \n", epsMem)

            # Remove the cost element from these episodes
            topK = [x[0] for x in epsMem]
            topK = torch.Tensor(topK)

            # Calculate the new mean and covariance
            mu = torch.mean(topK, axis = 0)
            std = torch.std(topK, axis = 0)
            var = torch.square(std)
            noise = torch.Tensor([0.2]*Horizon*len(ACTIVE_JOINTS))
            var = var + noise
            cov = torch.Tensor(np.diag(var))
            currEpsNum = Episodes - TopKEps

        # Extract the best action
        bestAction = epsMem[0][0][0:len(ACTIVE_JOINTS)]
        saveAction.append(bestAction)
        
        # List to store 1 state-action-state tuple
        pairs = []

        # Saving the previous state
        pairs.extend(getConfig(uid, ACTIVE_JOINTS))
        jointPositions = getJointPos(uid)
        for pos in jointPositions:
            pairs.extend(pos)
        pairs.extend(getState(uid)[1])
        pairs.extend(bestAction)
        
        applyAction(uid, bestAction) # Apply the best action to advance the robot
        # Calculate new distance to goal
        distToGoal = dist(torch.Tensor(initCoords), torch.Tensor(goalCoords))

        # Save and restore state to avoid pybullet syncing issue
        temp = p.saveState()
        p.restoreState(temp)

        # Save the action
        pairs.extend(getConfig(uid, ACTIVE_JOINTS))
        
        # Save the next state
        jointPositions = getJointPos(uid)
        for pos in jointPositions:
            pairs.extend(pos)
        pairs.extend(getState(uid)[1])
        saveRun.append(pairs)

        # Save the time taken by this iteration
        iterationTime = time.time() - startTime
        iterationTimes.append(iterationTime)
        
        # Save the end-effector position
        eePos = getState(uid)[0]
        finalEePos.append(eePos.tolist())

        # Calculate how far still the end-effector is from goal
        distError = dist(eePos, goalCoords)
        print(f"\neePos: \n{eePos}, \n\ngoalCoords: \n{goalCoords}, \n\nnextDistError: \n{distError}")
        
        # Check to see if the error is within threshold
        if distError < 0.02:
            print(f"reached position: \n{eePos}, \nwith target:\n{goalCoords}, \nand distError: \n{distError} \nin iteration {envStep}")
            break

    # Save all end-effector positions
    finalEePos = np.array(finalEePos)
    
    # traj = np.array(traj)

    # Calculate the average time required for each iteration
    avgIterationTime = np.average(iterationTimes)

    pathNum = sys.argv[1] # Path number from the command line argument

    # Write all necessary data to files
    with open(trainingFolder + f"ur5sample_{pathNum}.pkl", 'wb') as f:
        pickle.dump(saveRun, f)

    with open(errorFolder + f"debug_{pathNum}.pkl", 'wb') as f:
        pickle.dump(debug, f)

    with open(errorFolder + f"finalEePos_{pathNum}.pkl", 'wb') as f:
        pickle.dump(finalEePos, f)

    # with open(errorFolder + f"traj_{pathNum}.pkl", 'wb') as f:
    #     pickle.dump(traj, f)

    p.disconnect()
    
    # while 1:
    #     p.stepSimulation()
    
    # Calculate total time taken
    totalDuration = time.time() - initialTime
    print("avgIterationTime: ", time.strftime('%H:%M:%S', time.gmtime(avgIterationTime)))
    print("totalDuration: ", time.strftime('%H:%M:%S', time.gmtime(totalDuration)))

if __name__ == '__main__':
    main()
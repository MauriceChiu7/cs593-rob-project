import os
import pybullet as p
import pybullet_data
import torch
import pickle
import time
import argparse

ACTIVE_JOINTS = [1,2,3,4,5,6,8,9]
END_EFFECTOR_INDEX = 7 # The end effector link index.
CTL_FREQ = 20
SIM_STEPS = 3


def diff(v1, v2):
    '''
    Description: Calculates the difference between two vectors.

    Input:
    :v1 - {torch.Tensor} The first vector.
    :v2 - {torch.Tensor} The second vector.

    Returns:
    :diff - {torch.Tensor} The difference between the two vectors.
    '''
    return torch.sub(v1, v2)

"""
Calculates the magnitude of a vector.
"""
def magnitude(v):
    '''
    Description: Calculates the magnitude of a vector.

    Input:
    :v - {torch.Tensor} The vector.

    Returns:
    :magnitude - {float} The magnitude of the vector.

    '''
    return torch.sqrt(torch.sum(torch.pow(v, 2)))


def dist(p1, p2):
    '''
    Description: Calculates the distance between two points.

    Input:
    :p1 - {torch.Tensor} The first point.
    :p2 - {torch.Tensor} The second point.

    Returns:
    :dist - {float} The distance between the two points.
    '''
    return magnitude(diff(p1, p2))


def loadEnv():
    '''
    Description: Loads pybullet environment with a horizontal plane and earth like gravity.
    
    Input: None

    Returns: None

    '''
    # p.connect(p.DIRECT) 
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0.1])
    # p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0])
    p.setGravity(0, 0, -9.8)
    # p.setTimeStep(1./50.)
    p.setTimeStep(1./CTL_FREQ/SIM_STEPS)


def loadUR5():
    '''
    Description: Loads UR5 robot.

    Input: None

    Returns: 
    :uid - {int} The unique id of the robot.

    '''
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "UR5/ur5.urdf"), [0, 0, 0.1])
    p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=-45, cameraPitch=-45, cameraTargetPosition=(0,0,0.1))
    # p.resetDebugVisualizerCamera(cameraDistance=0.02, cameraYaw=90, cameraPitch=-0.125, cameraTargetPosition=(0,0.25,0.1))
    path = f"{os.getcwd()}/../ur5pybullet"
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
    return uid

def getJointPos(uid):
    '''
    Description: Gets the position of all the joints.

    Input:
    :uid - {int} The unique id of the robot.

    Returns:
    :jointPositions - {torch.Tensor} The position of all the joints.
    '''
    jointStates = p.getLinkStates(uid, ACTIVE_JOINTS)
    jointPos = []
    for j in jointStates:
        x, y, z = j[0]
        jointPos.append([x, y, z])
    jointPos[1] = torch.sub(torch.Tensor(jointPos[1]),torch.mul(diff(torch.Tensor(jointPos[1]), torch.Tensor(jointPos[4])), 0.3)).tolist()
    return jointPos

def drawHeight(uid):
    '''
    Description: Draws the height of the robot.

    Input:
    :uid - {int} The unique id of the robot.

    Returns: None
    '''
    jointPositions = getJointPos(uid)
    for pos in jointPositions:
        p.addUserDebugLine([pos[0], pos[1], pos[2]], [pos[0], pos[1], 0.1], [0,1,0], lineWidth=10, lifeTime=0.1)

def applyAction(uid, action):
    '''
    Description: Applies the action to the robot.

    Input:
    :uid - {int} The unique id of the robot.
    :action - {torch.Tensor} The action to be applied.

    Returns: None
    '''
    p.setJointMotorControlArray(uid, ACTIVE_JOINTS, p.POSITION_CONTROL, action)
    maxSimSteps = 150
    for s in range(maxSimSteps):
        # drawHeight(uid)
        print(s)
        time.sleep(1./25)
        p.stepSimulation()
        currConfig = getConfig(uid, ACTIVE_JOINTS)[0:8]
        action = torch.Tensor(action)
        currConfig = torch.Tensor(currConfig)
        error = torch.sub(action, currConfig)
        print("error:\n", error)
        done = True
        for e in error:
            if abs(e) > 0.02:
                done = False
        if done:
            # print(f"reached position: \n{action}, \nwith target:\n{currConfig}, \nand error: \n{error} \nin step {s}")
            break

def getConfig(uid, jointIds):
    '''
    Description: Gets the configuration of the robot in radian.

    Input:
    :uid - {int} The unique id of the robot.
    :jointIds - {list} The list of joint ids.

    Returns:
    :config - {torch.Tensor} The configuration of the robot.
    '''
    jointPositions = []
    for id in jointIds:
        # print(p.getJointState(uid, id)[0])
        jointPositions.append(p.getJointState(uid, id)[0])
    jointPositions = torch.Tensor(jointPositions)
    return jointPositions

def moveTo(uid, position):
    '''
    Description: Moves the robot to the given position.

    Input:
    :uid - {int} The unique id of the robot.
    :position - {torch.Tensor} The position to move to.

    Returns:
    :initState - {torch.Tensor} The initial state of the robot.
    :initCoords - {torch.Tensor} The initial coordinates of the robot.
    '''
    applyAction(uid, position)
    initState = getConfig(uid, ACTIVE_JOINTS)
    initCoords = torch.Tensor(p.getLinkState(uid, END_EFFECTOR_INDEX, 1)[0])
    # p.addUserDebugLine([0,0,0.1], initCoords, [1,0,0])
    # p.addUserDebugText("Replaying", [0.2, 0.2, 0], [0, 0, 10])
    return initState, initCoords

def playback(args):
    '''
    Description: Plays back the recorded data.

    Input:
    :args - {list} The arguments to be passed to the function.

    Returns: None
    '''

    if args.mode == 'mpc':
        path = args.path_number
        # with open(f"./trainingDataWithEE/ur5sample.pkl", 'rb') as f:
        with open(f"./trainingDataWithEE/ur5sample_{path}.pkl", 'rb') as f:
            tuples = pickle.load(f)
        
        # with open(f"./error/debug.pkl", 'rb') as f:
        with open(f"./error/debug_{path}.pkl", 'rb') as f:
            states = pickle.load(f)
    else:
        with open(f"./testRunResults/test.pkl", 'rb') as f:
            tuples = pickle.load(f)


    loadEnv()
    
    uid = loadUR5()
    
    # while 1:
    #     drawHeight(uid)
    #     p.stepSimulation()

    # goalCoords       tensor([-0.0587, -0.2683,  0.2389])
    # initState        tensor([-0.5888,  0.1401, -2.1266,  0.5458, -1.2973,  1.6745, -0.8886, -0.5362])
    # initCoords       tensor([-0.3674,  0.0948,  0.3818])
    if args.mode == 'mpc':
        initCoords = states["initCoords"]
        position = states["initState"][0:8]
        goalCoords = states["goalCoords"]

        print("position\t", position)

        p.addUserDebugLine([0,0,0.1], initCoords, [1,0,0])
        p.addUserDebugLine([0,0,0.1], goalCoords, [0,0,1])
        
        replay_initState, replay_initCoords = moveTo(uid, position)
        print("initCoords\t", initCoords)
        print("replay_initCoords\t", replay_initCoords)
    else:
        initCoords = [-0.8144, -0.1902,  0.1]
        goalCoords = [-0.6484, -0.3258,  0.3040]
        p.addUserDebugLine([0,0,0.1], goalCoords, [0,0,1])
        p.addUserDebugLine([0,0,0.1], initCoords, [1,0,0])


    
    

    # while 1:
    #     p.stepSimulation()
    
    # print("var\t", var)
    # print("var\t", var)

    if args.mode == 'mpc':
        for tuple in tuples:
            action = tuple[11:19]
            action = torch.Tensor(action)
            # print(f"action applied:\n{action}")
            applyAction(uid, action)
    else: 
        for tuple in tuples:
            action = torch.Tensor(tuple)
            applyAction(uid, action)
        


    # while 1:
    #     p.stepSimulation()
    #     drawHeight(uid)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for playing back actions for the ur5')

    parser.add_argument('-m', '--mode', type=str, default='mpc', choices=['mpc', 'nnmpc'], help="use 'mpc' to playback results generated with ur5.py and 'nnmpc' for results generated with nnur5mpc.py")
    parser.add_argument('-pn', '--path-number', type=int, default=999, help="the path number which you want to see the playback of")
    args = parser.parse_args()
    playback(args)
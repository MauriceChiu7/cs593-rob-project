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

"""
Loads pybullet environment with a horizontal plane and earth like gravity.
"""
def loadEnv():
    # p.connect(p.DIRECT) 
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0.1])
    p.setGravity(0, 0, -9.8)
    # p.setTimeStep(1./50.)
    p.setTimeStep(1./CTL_FREQ/SIM_STEPS)

"""
Loads the UR5 robot.
"""
def loadUR5():
    p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=-70, cameraPitch=-10, cameraTargetPosition=(0,0,0))
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


def applyAction(uid, jointIds, action):
    action = torch.Tensor(action)
    # action = torch.mul(torch.Tensor([1,1,1,1,1,1,0,0]))
    # print(f"action applied:\n{action}")
    p.setJointMotorControlArray(uid, jointIds, p.POSITION_CONTROL, action)
    for _ in range(SIM_STEPS):
        time.sleep(1./25.)
        p.stepSimulation()

def getConfig(uid, jointIds):
    jointPositions = []
    for id in jointIds:
        # print(p.getJointState(uid, id)[0])
        jointPositions.append(p.getJointState(uid, id)[0])
    jointPositions = torch.Tensor(jointPositions)
    return jointPositions

def moveTo(uid, position):
    p.setJointMotorControlArray(uid, ACTIVE_JOINTS, p.POSITION_CONTROL, position)
    # Give it some time to move there
    # for _ in range(100):
    for _ in range(SIM_STEPS):
        time.sleep(1./25.)
        p.stepSimulation()
    initState = getConfig(uid, ACTIVE_JOINTS)
    initCoords = torch.Tensor(p.getLinkState(uid, END_EFFECTOR_INDEX, 1)[0])
    p.addUserDebugLine([0,0,0.1], initCoords, [1,0,0])
    # p.addUserDebugText("Replaying", [0.2, 0.2, 0], [0, 0, 10])
    return initState, initCoords

def playback(args):

    if args.mode == 'mpc':
        path = args.path_number
        with open(f"./trainingDataWithEE/ur5sample_{path}.pkl", 'rb') as f:
            tuples = pickle.load(f)
        
        with open(f"./error/debug_{path}.pkl", 'rb') as f:
            states = pickle.load(f)
    else:
        with open(f"./testRunResults/test.pkl", 'rb') as f:
            tuples = pickle.load(f)


    loadEnv()
    
    uid = loadUR5()
    
    # goalCoords       tensor([-0.0587, -0.2683,  0.2389])
    # initState        tensor([-0.5888,  0.1401, -2.1266,  0.5458, -1.2973,  1.6745, -0.8886, -0.5362])
    # initCoords       tensor([-0.3674,  0.0948,  0.3818])
    if args.mode == 'mpc':
        initCoords = states["initCoords"]
        position = states["initState"]
        goalCoords = states["goalCoords"]

        replay_initState, replay_initCoords = moveTo(uid, position)

        print("initCoords\t", initCoords)
        print("replay_initCoords\t", replay_initCoords)
    else:
        initCoords = [-0.8144, -0.1902,  0.1]
        goalCoords = [-0.6484, -0.3258,  0.3040]


    p.addUserDebugLine([0,0,0.1], initCoords, [1,0,0])
    p.addUserDebugLine([0,0,0.1], goalCoords, [0,0,1])

    # while 1:
    #     p.stepSimulation()
    
    # print("var\t", var)
    # print("var\t", var)

    if args.mode == 'mpc':
        for tuple in tuples:
            action = tuple[11:19]
            action = torch.Tensor(action)
            # print(f"action applied:\n{action}")
            p.setJointMotorControlArray(uid, ACTIVE_JOINTS, p.POSITION_CONTROL, action)
            for _ in range(SIM_STEPS):
                time.sleep(1./25.)
                p.stepSimulation()
    else: 
        for tuple in tuples:
            action = torch.Tensor(tuple)
            p.setJointMotorControlArray(uid, ACTIVE_JOINTS, p.POSITION_CONTROL, action)
            for _ in range(SIM_STEPS):
                time.sleep(1./25.)
                p.stepSimulation()
        # time.sleep(1.)
    # time.sleep(10)
    # while 1:
    #     p.stepSimulation()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for playing back actions for the ur5')

    parser.add_argument('-m', '--mode', type=str, default='mpc', choices=['mpc', 'nnmpc'], help="use 'mpc' to playback results generated with ur5.py and 'nnmpc' for results generated with nnur5mpc.py")
    parser.add_argument('-pn', '--path-number', type=int, default=70, help="the path number which you want to see the playback of")
    args = parser.parse_args()
    playback(args)
import os
import pybullet as p
import pybullet_data
import torch
import pickle
import time

"""
Loads pybullet environment with a horizontal plane and earth like gravity.
"""
def loadEnv():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0.1])
    
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1./500)
    # p.setRealTimeSimulation(1)

"""
Loads the UR5 robot.
"""
def loadUR5(activeJoints):
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
                p.setCollisionFilterPair(uid, uid, l1, l0, enableCollision)
    return uid

"""
Applys a random action to the all the joints.
"""
def applyAction(uid, jointIds, action):
    action = torch.Tensor(action)
    print(f"action applied:\n{action}")
    p.setJointMotorControlArray(uid, jointIds, p.POSITION_CONTROL, action)
    p.stepSimulation()

def playback():
    with open('./trainingData/ur5/ur5sample.pkl', 'rb') as f:
        tuples = pickle.load(f)
    
    loadEnv()
    
    ACTIVE_JOINTS = [1,2,3,4,5,6,8,9]
    uid = loadUR5(ACTIVE_JOINTS)
    
    # moveToStartingPose(uid, ACTIVE_JOINTS)

    for tuple in tuples:
        action = tuple[8:16]
        print(action)
        applyAction(uid, ACTIVE_JOINTS, action)
        time.sleep(1./25.)
        # time.sleep(1.)

if __name__ == '__main__':
    
    playback()
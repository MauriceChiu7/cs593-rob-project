import pybullet as p
import time
import os
import pybullet_data
from ur5pybullet import ur5
from ur5util import *

END_EFFECTOR_INDEX = 7 # The end effector link index
# JOINT_TYPE_LIST = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]

def main():
    ## Creates an instance of the physics server
    p.connect(p.GUI)

    ## Loads a plane into the environment
    # plane = p.loadURDF("plane.urdf") # This can be used if you are working in the same directory as your pybullet library
    plane = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0.1]) # Otherwise, this is needed

    ## Setting up the environment
    p.setGravity(0, 0, -9.8)

    urdfFlags = p.URDF_USE_SELF_COLLISION

    ## Loads the UR5 into the environment
    path = f"{os.getcwd()}/ur5pybullet"
    os.chdir(path) # Needed to change directory to load the UR5

    handy = p.loadURDF(os.path.join(os.getcwd(), "./urdf/real_arm.urdf"), [0.0,0.0,0.0], p.getQuaternionFromEuler([0,0,0]), flags = p.URDF_USE_INERTIA_FROM_FILE)

    # Enable collision for all the link pairs.
    for l0 in range(p.getNumJoints(handy)):
        for l1 in range(p.getNumJoints(handy)):
            if (not l1>l0):
                enableCollision = 1
                # print("collision for pair",l0,l1, p.getJointInfo(handy,l0)[12],p.getJointInfo(handy,l1)[12], "enabled=",enableCollision)
                p.setCollisionFilterPair(handy, handy, l1, l0, enableCollision)

    
    linkStates = getLinkState(p, handy, END_EFFECTOR_INDEX, 1, verbose=False)
    (jointInfo, JointState) = getJointsInfo(p, handy, verbose=False)
    

    while(1):
        # Looping the simulation
        p.stepSimulation()
        time.sleep(1./500.)


if __name__ == "__main__":
    main()
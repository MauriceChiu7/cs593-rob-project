import pybullet as p
import time
import os
import pybullet_data
from ur5pybullet import ur5

"""
member functions:
__init__

Accessors: 
getActionDimension()
getObservationDimension()
getObservation()

Mutators:
reset()
resetJointPoses()
setPosition(pos, quat)
action(motorCommands)
move_to(position_delta, mode='abs', noise=False, clip=False)
"""

## Creates an instance of the physics server
p.connect(p.GUI)

## Loads a plane into the environment
# plane = p.loadURDF("plane.urdf") # This alternative can be used if you are working in the same directory as your pybullet library
plane = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, -1]) # Otherwise, this is needed

## Setting up the environment
p.setGravity(0, 0, -9.8)
# p.setTimeStep(1./500.)

urdfFlags = p.URDF_USE_SELF_COLLISION

## Loads the UR5 into the environment
# Needed to change directory to load the UR5
path = f"{os.getcwd()}/ur5pybullet"
os.chdir(path)
handy = ur5.ur5() # Creating an instance of the UR5

## Getting UR5's state information
# obs = handy.getObservation() # This contains all the states of the UR5
# print(f"obs: {obs}")

# obsDim = handy.getObservationDimension()
# print(f"obsDim: {obsDim}")

# actionDim = handy.getActionDimension()
# print(f"actionDim: {actionDim}")

handy.resetJointPoses() # This sets the UR5 to a "ready pose"
# handy.reset() # This resets the UR5 with all initial default values

# ur5.setup_sisbot / is a weird method and not sure what it does yet.

def getEnvInfo(uid):
    envInfo = p.getBasePositionAndOrientation(uid)
    print(f"Position: {envInfo[0]}")
    print(f"Orientation: {envInfo[1]}")
    return (envInfo[0], envInfo[1])

print("\n================================\n")
print(f"Environment information: ")
getEnvInfo(handy.uid)
print("\n================================\n")

def getLinkState(uid, linkIndex, computeLinkVelocity):
    linkState = p.getLinkState(uid, linkIndex, computeLinkVelocity)
    print(f"linkWorldPosition:              {linkState[0]}") # Cartesian position of center of mass
    print(f"linkWorldOrientation:           {linkState[1]}") # Cartesian orientation of center of mass, in quaternion [x, y, z, w]
    print(f"localInertialFramePosition:     {linkState[2]}") # local position offset of inertial frame (center of mass) expressed in the URDF link frame
    print(f"localInertialFrameOrientation:  {linkState[3]}") # local orientation (quaternion [x, y, z, w]) offset of the inertial frame expressed in URDF link frame
    print(f"worldLinkFramePosition:         {linkState[4]}") # world position of the URDF link frame
    print(f"worldLinkFrameOrientation:      {linkState[5]}") # world orientation of the URDF link frame
    print(f"worldLinkLinearVelocity:        {linkState[6]}") # Cartesian world linear velocity. Only returned if computeLinkVelocity non-zero.
    print(f"worldLinkAngularVelocity:       {linkState[7]}") # Cartesian world angular velocity. Only returned if computeLinkVelocity non-zero.
    return linkState

print("End Effector Link State:")
getLinkState(handy.uid, handy.endEffectorIndex, 1)
print("\n================================\n")

def getJointStates(uid, jointIndices):
    jointStates = p.getJointStates(uid, jointIndices)
    jointID = 0
    for joint in jointStates:
        print(f"Joint ID:       {jointID}")
        print(f"jointPosition: {joint[0]}")
        print(f"jointVelocity: {joint[1]}")
        print(f"jointReactionForces: {joint[2]}")
        print(f"appliedJointMotorTorque: {joint[3]}\n")
        jointID += 1
    return jointStates

print("Joint States: ")
getJointStates(handy.uid, handy.active_joint_ids)
print("================================\n")

while(1):
    # Looping the simulation
    p.stepSimulation()
    time.sleep(1./500.)

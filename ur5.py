import pybullet as p
import time
import os
import pybullet_data
from ur5pybullet import ur5

END_EFFECTOR_INDEX = 7 # The end effector link index
# JOINT_TYPE_LIST = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]

## Creates an instance of the physics server
p.connect(p.GUI)

## Loads a plane into the environment
# plane = p.loadURDF("plane.urdf") # This alternative can be used if you are working in the same directory as your pybullet library
plane = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), [0, 0, 0.1]) # Otherwise, this is needed

## Setting up the environment
p.setGravity(0, 0, -9.8)
# p.setTimeStep(1./500.)

urdfFlags = p.URDF_USE_SELF_COLLISION

## Loads the UR5 into the environment
path = f"{os.getcwd()}/ur5pybullet"
os.chdir(path) # Needed to change directory to load the UR5
# handy = ur5.ur5() # Creating an instance of the UR5
handy = p.loadURDF(os.path.join(os.getcwd(), "./urdf/real_arm.urdf"), [0.0,0.0,0.0], p.getQuaternionFromEuler([0,0,0]), flags=p.URDF_USE_INERTIA_FROM_FILE)

for l0 in range(p.getNumJoints(handy)):
    for l1 in range(p.getNumJoints(handy)):
        if (not l1>l0):
            enableCollision = 1
            # print("collision for pair",l0,l1, p.getJointInfo(handy,l0)[12],p.getJointInfo(handy,l1)[12], "enabled=",enableCollision)
            p.setCollisionFilterPair(handy, handy, l1, l0, enableCollision)

## Getting UR5's state information
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

print("\n=== Environment Info (End Effector Link State) ===\n")
print('The "linkWorldPosition" is the x, y, and z position of the UR5\'s end effector.')
print('The "linkWorldOrientation" is the orientation of the UR5\'s end effector.\n')
getLinkState(handy, END_EFFECTOR_INDEX, 1)
print("\n================ Joint Info ================\n")


def getJointsInfo(uid):
    for joint in range(p.getNumJoints(uid)):
        info = p.getJointInfo(uid, joint)
        state = p.getJointState(uid, joint)
        print(f"jointIndex:                 {info[0]}")
        print(f"jointName:                  {info[1]}")
        print(f"jointPosition:              {state[0]}")
        print(f"jointVelocity:              {state[1]}")
        print(f"jointReactionForces:        {state[2]}")
        print(f"appliedJointMotorTorque:    {state[3]}")
        print(f"jointType:                  {info[2]}")
        print(f"qIndex:                     {info[3]}")
        print(f"uIndex:                     {info[4]}")
        print(f"flags:                      {info[5]}")
        print(f"jointDamping:               {info[6]}")
        print(f"jointFriction:              {info[7]}")
        print(f"jointLowerLimit:            {info[8]}")
        print(f"jointUpperLimit:            {info[9]}")
        print(f"jointMaxForce:              {info[10]}")
        print(f"jointMaxVelocity:           {info[11]}")
        print(f"linkName:                   {info[12]}")
        print(f"jointAxis:                  {info[13]}")
        print(f"parentFramePos:             {info[14]}")
        print(f"parentFrameOrn:             {info[15]}")
        print(f"parentIndex:                {info[15]}\n")
        # if info[2]==1: # set revolute joint to static
        #     p.setJointMotorControl2(uid, info[0], p.VELOCITY_CONTROL, targetVelocity=0, force=0)


print("Joint Info: ")
getJointsInfo(handy)
print("================================\n")

while(1):
    # Looping the simulation
    p.stepSimulation()
    time.sleep(1./500.)

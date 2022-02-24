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
plane = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), 0, 0, -1) # Otherwise, this is needed

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
obs = handy.getObservation() # This contains all the states of the UR5
print(f"obs: {obs}")

obsDim = handy.getObservationDimension()
print(f"obsDim: {obsDim}")

actionDim = handy.getActionDimension()
print(f"actionDim: {actionDim}")

handy.resetJointPoses() # This sets the UR5 to a "ready pose"
# handy.reset() # This resets the UR5 with all initial default values

# ur5.setup_sisbot / is a weird method and not sure what it does yet.

while(1):
    # Looping the simulation
    p.stepSimulation()
    time.sleep(1./500.)

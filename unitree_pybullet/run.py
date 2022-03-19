import pybullet as p
import time
import csv
import numpy as np

# class Dog:
p.connect(p.GUI)
plane = p.loadURDF("plane.urdf")
p.setGravity(0,0,-9.8)
p.setTimeStep(1./500)
#p.setDefaultContactERP(0)
#urdfFlags = p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
urdfFlags = p.URDF_USE_SELF_COLLISION
quadruped = p.loadURDF("a1/urdf/a1.urdf",[0,0,0.48],[0,0,0,1], flags = urdfFlags,useFixedBase=False)

lower_legs = [2,5,8,11]
for l0 in lower_legs:
    for l1 in lower_legs:
        if (l1>l0):
            enableCollision = 1
            print("collision for pair",l0,l1, p.getJointInfo(quadruped,l0)[12],p.getJointInfo(quadruped,l1)[12], "enabled=",enableCollision)
            p.setCollisionFilterPair(quadruped, quadruped, 2,5,enableCollision)

jointIds=[]
paramIds=[]

maxForceId = p.addUserDebugParameter("maxForce",0,100,20)

for j in range (p.getNumJoints(quadruped)):
    p.changeDynamics(quadruped,j,linearDamping=0, angularDamping=0)
    info = p.getJointInfo(quadruped,j)
    # print(info)
    jointName = info[1]
    jointType = info[2]
    if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
        jointIds.append(j)

# print(jointIds)

p.getCameraImage(480,320)
p.setRealTimeSimulation(0)

joints=[]

with open("final_actions2.csv","r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        # print(row)
        strRow = np.array(row)
        fltRow = strRow.astype(float)
        p.setJointMotorControlArray(quadruped, jointIds, p.POSITION_CONTROL, fltRow)
        for _ in range(10):
            p.stepSimulation()
        # for lower_leg in lower_legs:
            #print("points for ", quadruped, " link: ", lower_leg)
            # pts = p.getContactPoints(quadruped,-1, lower_leg)
            #print("num points=",len(pts))
            #for pt in pts:
            #    print(pt[9])
        # time.sleep(2.)
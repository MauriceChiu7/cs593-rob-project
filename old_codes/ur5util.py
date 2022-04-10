ACTIVE_JOINTS = [1,2,3,4,5,6,8,9]

def getCurrJointsState(p, uid):
    jointsState = []
    for a in ACTIVE_JOINTS:
        jointsState.append(p.getJointState(uid, a))
    return jointsState

def getJointRange(p, uid):
    jointsRange = []
    for a in ACTIVE_JOINTS:
        jointInfo = p.getJointInfo(uid, a)
        jointsRange.append((jointInfo[8], jointInfo[9]))
    return jointsRange

# Gets link state
def getLinkState(p, uid, linkIndex, computeLinkVelocity, verbose=False):
    linkState = p.getLinkState(uid, linkIndex, computeLinkVelocity)
    if verbose: 
        print("\n=== Environment Info (End Effector Link State) ===\n")
        print('The "linkWorldPosition" is the x, y, and z position of the UR5\'s end effector.')
        print('The "linkWorldOrientation" is the orientation of the UR5\'s end effector.\n')
        print(f"linkWorldPosition:              {linkState[0]}") # Cartesian position of center of mass
        print(f"linkWorldOrientation:           {linkState[1]}") # Cartesian orientation of center of mass, in quaternion [x, y, z, w]
        print(f"localInertialFramePosition:     {linkState[2]}") # local position offset of inertial frame (center of mass) expressed in the URDF link frame
        print(f"localInertialFrameOrientation:  {linkState[3]}") # local orientation (quaternion [x, y, z, w]) offset of the inertial frame expressed in URDF link frame
        print(f"worldLinkFramePosition:         {linkState[4]}") # world position of the URDF link frame
        print(f"worldLinkFrameOrientation:      {linkState[5]}") # world orientation of the URDF link frame
        print(f"worldLinkLinearVelocity:        {linkState[6]}") # Cartesian world linear velocity. Only returned if computeLinkVelocity non-zero.
        print(f"worldLinkAngularVelocity:       {linkState[7]}") # Cartesian world angular velocity. Only returned if computeLinkVelocity non-zero.
        print("\n================================\n")
    return linkState

# Gets joint info
def getJointsInfo(p, uid, verbose=False):
    for joint in range(p.getNumJoints(uid)):
        info = p.getJointInfo(uid, joint)
        state = p.getJointState(uid, joint)
        if verbose:
            print(f"\n================ Joint {info[0]} Info ================\n")
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
            print(f"parentIndex:                {info[15]}")
            print("\n================================\n")
        # if info[2]==1: # set revolute joint to static
        #     p.setJointMotorControl2(uid, info[0], p.VELOCITY_CONTROL, targetVelocity=0, force=0)
    return (info, state)
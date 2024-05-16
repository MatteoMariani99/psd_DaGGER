import pybullet as p
import time
import math

p.connect(p.GUI)

p.resetSimulation()

p.setGravity(0,0,-10)
useRealTimeSim = 0

p.setRealTimeSimulation(1)

#track = p.loadURDF("plane.urdf")
track = p.loadSDF("f10_racecar/meshes/barca_track.sdf", globalScaling=1)
#otherCar = p.loadURDF("f10_racecar/racecar_differential.urdf", [0,1,.3])
car = p.loadURDF("f10_racecar/racecar_differential.urdf", [0,0,0.3])

p.setGravity(0,0,-10)

    
# for wheel in range(p.getNumJoints(car)):
#     print("joint[",wheel,"]=", p.getJointInfo(car,wheel))
#     p.setJointMotorControl2(car,wheel,p.VELOCITY_CONTROL,targetVelocity=0,force=0)
#     p.getJointInfo(car,wheel)	

# wheels = [8,15]
# print("----------------")

p.setJointMotorControl2(car,10,p.VELOCITY_CONTROL,targetVelocity=1,force=10)
c = p.createConstraint(car,9,car,11,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
p.changeConstraint(c,gearRatio=1, maxForce=10000)

c = p.createConstraint(car,10,car,13,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
p.changeConstraint(c,gearRatio=-1, maxForce=10000)

c = p.createConstraint(car,9,car,13,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
p.changeConstraint(c,gearRatio=-1, maxForce=10000)

c = p.createConstraint(car,16,car,18,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
p.changeConstraint(c,gearRatio=1, maxForce=10000)


c = p.createConstraint(car,16,car,19,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
p.changeConstraint(c,gearRatio=-1, maxForce=10000)

c = p.createConstraint(car,17,car,19,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
p.changeConstraint(c,gearRatio=-1, maxForce=10000)

c = p.createConstraint(car,1,car,18,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
p.changeConstraint(c,gearRatio=-1, gearAuxLink = 15, maxForce=10000)
c = p.createConstraint(car,3,car,19,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
p.changeConstraint(c,gearRatio=-1, gearAuxLink = 15,maxForce=10000)


steering = [0,2]

hokuyo_joint=4
zed_camera_joint = 5


frame = 0

# lineId = p.addUserDebugLine([0,0,0],[0,0,1],[1,0,0])
# lineId2 = p.addUserDebugLine([0,0,0],[0,0,1],[1,0,0])
# lineId3= p.addUserDebugLine([0,0,0],[0,0,1],[1,0,0])
# print("lineId=",lineId)

camInfo = p.getDebugVisualizerCamera()
lastTime = time.time()
frame=0
leftWheelVelocity=0
rightWheelVelocity=0
forward=0
turn = 0
for j in range (p.getNumJoints(car)):
	print(p.getJointInfo(car,j))

while (True):
    ls = p.getLinkState(car,zed_camera_joint, computeForwardKinematics=True)
    camPos = ls[0]
    camOrn = ls[1]
    camMat = p.getMatrixFromQuaternion(camOrn)
    upVector = [0,0,1]
    forwardVec = [camMat[0],camMat[3],camMat[6]]
    #sideVec =  [camMat[1],camMat[4],camMat[7]]
    camUpVec =  [camMat[2],camMat[5],camMat[8]]
    camTarget = [camPos[0]+forwardVec[0]*10,camPos[1]+forwardVec[1]*10,camPos[2]+forwardVec[2]*10]
    camUpTarget = [camPos[0]+camUpVec[0],camPos[1]+camUpVec[1],camPos[2]+camUpVec[2]]
    viewMat = p.computeViewMatrix(camPos, camTarget, camUpVec)
    projMat = camInfo[3]
    #p.getCameraImage(320,200,viewMatrix=viewMat,projectionMatrix=projMat, flags=p.ER_NO_SEGMENTATION_MASK, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    p.getCameraImage(320,200,viewMatrix=viewMat,projectionMatrix=projMat, renderer=p.ER_BULLET_HARDWARE_OPENGL)

    
    time.sleep(1./240.)
    keys = p.getKeyboardEvents()
    
    for k,v in keys.items():
        if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                turn = 0.8
        if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_RELEASED)):
                turn = 0
        if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                turn = -0.8
        if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_RELEASED)):
                turn = 0

        if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                forward=20
        if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_RELEASED)):
                forward=0
        if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                forward=-20
        if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_RELEASED)):
                forward=0

    #print("f:",forward)
    #print(turn)
    p.setJointMotorControl2(car,1,p.VELOCITY_CONTROL,targetVelocity=forward)
    p.setJointMotorControl2(car,3,p.VELOCITY_CONTROL,targetVelocity=forward)
    p.setJointMotorControl2(car,12,p.VELOCITY_CONTROL,targetVelocity=forward)
    p.setJointMotorControl2(car,14,p.VELOCITY_CONTROL,targetVelocity=forward)
    p.setJointMotorControl2(car,0,p.POSITION_CONTROL,targetPosition=-turn)
    p.setJointMotorControl2(car,2,p.POSITION_CONTROL,targetPosition=-turn)
        


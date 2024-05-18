import pybullet as p
import time
import pybullet_data
import numpy as np
import pandas as pd
from environment import PyBulletContinuousEnv
import model_new
import cv2
import torch

vel_max = 15

def getCamera_image():
        camInfo = p.getDebugVisualizerCamera()
        ls = p.getLinkState(turtle,7, computeForwardKinematics=True)
        camPos = ls[0]
        camOrn = ls[1]
        camMat = p.getMatrixFromQuaternion(camOrn)
        #upVector = [0,0,1]
        forwardVec = [camMat[0],camMat[3],camMat[6]]
        #sideVec =  [camMat[1],camMat[4],camMat[7]]
        camUpVec =  [camMat[2],camMat[5],camMat[8]]
        camTarget = [camPos[0]+forwardVec[0]*10,camPos[1]+forwardVec[1]*10,camPos[2]+forwardVec[2]*10]
        #camUpTarget = [camPos[0]+camUpVec[0],camPos[1]+camUpVec[1],camPos[2]+camUpVec[2]]
        viewMat = p.computeViewMatrix(camPos, camTarget, camUpVec)
        projMat = camInfo[3]
        
        # ottengo le 3 immagini: rgb, depth, segmentation
        width, height, rgbImg, depthImg, segImg= p.getCameraImage(200,66,viewMatrix=viewMat,projectionMatrix=projMat, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        # faccio un reshape in quanto da sopra ottengo un array di elementi
        rgb_opengl = np.reshape(rgbImg, (height, width, 4)) 

        # Tolgo il canale alpha e converto da BGRA a RGB per la rete
        rgb_image = cv2.cvtColor(rgb_opengl, cv2.COLOR_BGRA2RGB)
        #print("Reshaped RGB image size:", rgb_image.shape)

        return rgb_image

model = model_new.VehicleControlModel(vel_max)

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetDebugVisualizerCamera(cameraDistance=15, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[15,0,0])
offset = [0,0,0]
p.loadURDF("plane.urdf")
#turtle = p.loadURDF("f10_racecar/simplecar.urdf", [0,0,.3])
#turtle = p.loadURDF("f10_racecar/simplecar.urdf", [-15,-11,.3])
#track = p.loadSDF("f10_racecar/meshes/barca_track.sdf", globalScaling=1)

for i in range(0,30,1):
        p.loadURDF("cube/marble_cube.urdf",[i,-1.5,0],useFixedBase = True)
for l in range(0,30,1):
        p.loadURDF("cube/marble_cube.urdf",[l,1.5,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[29,0,0],useFixedBase = True)
       
#turtle = p.loadURDF("f10_racecar/simplecar.urdf", [0,0,.3])
turtle = p.loadURDF("f10_racecar/simplecar.urdf", [0,0,.3])
sphere_color = [1, 0, 0,1]  # Red color
sphere_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=0)
sphere_body = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=sphere_shape,basePosition=[-10,5, 0])
p.changeVisualShape(sphere_body, -1, rgbaColor=sphere_color)
#track = p.loadSDF("f10_racecar/meshes/barca_track.sdf", globalScaling=1)
p.setRealTimeSimulation(1)

image = getCamera_image()
image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
image = np.transpose(image, (2, 0, 1))
#print(image.shape)
#print(image[np.newaxis,...].shape)
prediction = model(torch.from_numpy(image[np.newaxis,...]).type(torch.FloatTensor))

#print(prediction)


# sphere_color = [1, 0, 0,1]  # Red color
# sphere_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=0)
# for i in range (1,10):
#         p.createMultiBody(baseMass=1, baseCollisionShapeIndex=sphere_shape,basePosition=[1,i,2])
#p.changeVisualShape(sphere_body, -1, rgbaColor=sphere_color)

# for j in range (p.getNumJoints(turtle)):
# 	print(p.getJointInfo(turtle,j))
forward=0
turn=0
while (1):
        p.setGravity(0,0,-10)
        time.sleep(1./240.)
        keys = p.getKeyboardEvents()
        leftWheelVelocity=0
        rightWheelVelocity=0
        speed=10
	
        for k,v in keys.items():

                if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                        turn = -0.5
                if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_RELEASED)):
                        turn = 0
                if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                        turn = 0.5
                if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_RELEASED)):
                        turn = 0

                if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                        forward=vel_max
                if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_RELEASED)):
                        forward=0
                if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                        forward=-vel_max
                if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_RELEASED)):
                        forward=0

        
        # F10 RACECAR
        # p.setJointMotorControl2(turtle,1,p.VELOCITY_CONTROL,targetVelocity=forward)
        # p.setJointMotorControl2(turtle,3,p.VELOCITY_CONTROL,targetVelocity=forward)
        # p.setJointMotorControl2(turtle,12,p.VELOCITY_CONTROL,targetVelocity=forward)
        # p.setJointMotorControl2(turtle,14,p.VELOCITY_CONTROL,targetVelocity=forward)
        # p.setJointMotorControl2(turtle,0,p.POSITION_CONTROL,targetPosition=-turn)
        # p.setJointMotorControl2(turtle,2,p.POSITION_CONTROL,targetPosition=-turn)

        # SIMPLE CAR
        # turn positivo giro a sinistra
        # turn negativo giro a destra
        p.setJointMotorControl2(turtle,0,p.POSITION_CONTROL,targetPosition=turn)
        p.setJointMotorControl2(turtle,2,p.POSITION_CONTROL,targetPosition=turn)
        p.setJointMotorControl2(turtle,1,p.VELOCITY_CONTROL,targetVelocity=forward)
        p.setJointMotorControl2(turtle,3,p.VELOCITY_CONTROL,targetVelocity=forward)
        p.setJointMotorControl2(turtle,4,p.VELOCITY_CONTROL,targetVelocity=forward)
        p.setJointMotorControl2(turtle,5,p.VELOCITY_CONTROL,targetVelocity=forward)




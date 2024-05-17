import pybullet as p
import time
import pybullet_data
import numpy as np
import pandas as pd

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[0,0,0])
offset = [0,0,0]
p.loadURDF("plane.urdf")
#turtle = p.loadURDF("f10_racecar/simplecar.urdf", [0,0,.3])
#turtle = p.loadURDF("f10_racecar/simplecar.urdf", [-15,-11,.3])
#track = p.loadSDF("f10_racecar/meshes/barca_track.sdf", globalScaling=1)
p.loadURDF("cube/marble_cube.urdf",[-3,2,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[-3,1,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[-3,3,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[-3,4,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[-3,0,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[-3,-1,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[-3,-2,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[-3,-3,0],useFixedBase = True)

# horizontal
p.loadURDF("cube/marble_cube.urdf",[3,5,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[2,5,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[4,5,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[1,5,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[5,5,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[6,5,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[7,5,0],useFixedBase = True)

# vertical
p.loadURDF("cube/marble_cube.urdf",[8,7,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[8,6,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[8,8,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[8,5,0],useFixedBase = True)

# vertical
p.loadURDF("cube/marble_cube.urdf",[-8,7,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[-8,6,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[-8,8,0],useFixedBase = True)
p.loadURDF("cube/marble_cube.urdf",[-8,5,0],useFixedBase = True)

#turtle = p.loadURDF("f10_racecar/simplecar.urdf", [0,0,.3])
turtle = p.loadURDF("f10_racecar/simplecar.urdf", [0,0,.3])
sphere_color = [1, 0, 0,1]  # Red color
sphere_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=0)
sphere_body = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=sphere_shape,basePosition=[-10,5, 0])
p.changeVisualShape(sphere_body, -1, rgbaColor=sphere_color)
#track = p.loadSDF("f10_racecar/meshes/barca_track.sdf", globalScaling=1)
p.setRealTimeSimulation(1)



# sphere_color = [1, 0, 0,1]  # Red color
# sphere_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=0)
# for i in range (1,10):
#         p.createMultiBody(baseMass=1, baseCollisionShapeIndex=sphere_shape,basePosition=[1,i,2])
#p.changeVisualShape(sphere_body, -1, rgbaColor=sphere_color)

for j in range (p.getNumJoints(turtle)):
	print(p.getJointInfo(turtle,j))
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
                        turn = 0.5
                if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_RELEASED)):
                        turn = 0
                if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                        turn = -0.5
                if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_RELEASED)):
                        turn = 0

                if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                        forward=15
                if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_RELEASED)):
                        forward=0
                if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                        forward=-15
                if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_RELEASED)):
                        forward=0

        #print(forward)ù
        # F10 RACECAR
        # p.setJointMotorControl2(turtle,1,p.VELOCITY_CONTROL,targetVelocity=forward)
        # p.setJointMotorControl2(turtle,3,p.VELOCITY_CONTROL,targetVelocity=forward)
        # p.setJointMotorControl2(turtle,12,p.VELOCITY_CONTROL,targetVelocity=forward)
        # p.setJointMotorControl2(turtle,14,p.VELOCITY_CONTROL,targetVelocity=forward)
        # p.setJointMotorControl2(turtle,0,p.POSITION_CONTROL,targetPosition=-turn)
        # p.setJointMotorControl2(turtle,2,p.POSITION_CONTROL,targetPosition=-turn)

        # SIMPLE CAR
        p.setJointMotorControl2(turtle,0,p.POSITION_CONTROL,targetPosition=-turn)
        p.setJointMotorControl2(turtle,2,p.POSITION_CONTROL,targetPosition=-turn)
        p.setJointMotorControl2(turtle,1,p.VELOCITY_CONTROL,targetVelocity=forward)
        p.setJointMotorControl2(turtle,3,p.VELOCITY_CONTROL,targetVelocity=forward)
        p.setJointMotorControl2(turtle,4,p.VELOCITY_CONTROL,targetVelocity=forward)
        p.setJointMotorControl2(turtle,5,p.VELOCITY_CONTROL,targetVelocity=forward)




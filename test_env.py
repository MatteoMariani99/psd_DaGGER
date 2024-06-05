import pybullet as p
import numpy as np
import time
import cv2

# Or directly
from environment import PyBulletContinuousEnv
env = PyBulletContinuousEnv()

# Inizializzo l'ambiente
#env.reset()

done = False
turn=0
forward=0
backward=0
p.resetDebugVisualizerCamera(cameraDistance=22, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[4,-2,0])
turtle = p.loadURDF("world&car/simplecar.urdf", [-10,1,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(180)]))
#turtle = p.loadURDF("f10_racecar/simplecar.urdf", [-12,-11,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(180)]))
#turtle = p.loadURDF("f10_racecar/simplecar.urdf", [-9,-6.5,.3])
#turtle = p.loadURDF("f10_racecar/simplecar.urdf", [21,0,.3],  p.getQuaternionFromEuler([0,0,np.deg2rad(-55)]))
#turtle = p.loadURDF("f10_racecar/simplecar.urdf", [35.5,2,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(-90)]))
#track = p.loadSDF("f10_racecar/meshes/prova.sdf")
#track = p.loadSDF("world/track_cones.sdf", globalScaling=1)
track = p.loadSDF("world&car/meshes/barca_track_modified.sdf", globalScaling=1)

# dopo 1000 step done diventa True
while not done:
	#time.sleep(1./240.)
	p.setGravity(0,0,-10)
	keys = p.getKeyboardEvents()

	# comandi da tastiera per l'expert
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
			forward=40
		if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_RELEASED)):
			forward=0
		if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_TRIGGERED)):
			backward=40
		if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_RELEASED)):
			backward=0

	p.setJointMotorControl2(turtle,0,p.POSITION_CONTROL,targetPosition=turn)
	p.setJointMotorControl2(turtle,2,p.POSITION_CONTROL,targetPosition=turn)
	p.setJointMotorControl2(turtle,1,p.VELOCITY_CONTROL,targetVelocity=forward)
	p.setJointMotorControl2(turtle,3,p.VELOCITY_CONTROL,targetVelocity=forward)
	p.setJointMotorControl2(turtle,4,p.VELOCITY_CONTROL,targetVelocity=forward)
	p.setJointMotorControl2(turtle,5,p.VELOCITY_CONTROL,targetVelocity=forward)



env.close()

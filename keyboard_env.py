import pybullet as p
import numpy as np
import time
import cv2

from environment_cones import PyBulletContinuousEnv
env = PyBulletContinuousEnv()


done = False
turn=0
forward=0
backward=0
p.resetDebugVisualizerCamera(cameraDistance=22, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[4,-2,0])
#p.loadURDF("world&car/plane/plane.urdf")
#turtle = p.loadURDF("world&car/simplecar.urdf", [-10,1,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(180)]))

#p.loadSDF("world/track.sdf",globalScaling = 1)
#turtle = p.loadURDF("f10_racecar/simplecar.urdf", [-12,-11,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(180)]))
#turtle = p.loadURDF("f10_racecar/simplecar.urdf", [-9,-6.5,.3])
#turtle = p.loadURDF("f10_racecar/simplecar.urdf", [21,0,.3],  p.getQuaternionFromEuler([0,0,np.deg2rad(-55)]))
#turtle = p.loadURDF("f10_racecar/simplecar.urdf", [35.5,2,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(-90)]))
#track = p.loadSDF("f10_racecar/meshes/prova.sdf")
#track = p.loadSDF("world/track_cones.sdf", globalScaling=1)
#track = p.loadSDF("world&car/meshes/barca_track_modified.sdf", globalScaling=1)
env.reset()
# dopo 1000 step done diventa True
while not done:
	start = time.time()
	#state= env.get_observation()
	color_rgb = env.getCamera_image()
        
	p.setGravity(0,0,-9.81)
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
			forward=10
		if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_RELEASED)):
			forward=0
		if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_TRIGGERED)):
			forward=-10
		if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_RELEASED)):
			forward=0
   
   

	#forward = 0
	image,_,_ = env.step([turn,forward])
	cv2.imshow("YOLOv8 Tracking", color_rgb)
	# Display the annotated frame
	#cv2.imshow("YOLOv8 detect", annotated_frame)
	#cv2.imshow('IMAGE', img)
	cv2.waitKey(1)
	

	print("-----seconds-----", time.time()-start)


env.close()

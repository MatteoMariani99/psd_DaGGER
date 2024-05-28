import pybullet as p
import time
import pybullet_data
import numpy as np
from environment import PyBulletContinuousEnv
import model_new
import cv2
import random
from get_track import *
from matplotlib import pyplot as plt

vel_max = 15
threshold = 0.25
#						          UL        UR          LR        LL
#roadPts4Homography = np.array([[214,355], [487,355], [632,472], [89,472] ]).astype(np.float32)
#roadDesiredPts     = np.array([[ 150,355], [500,355], [500,472], [150,472]  ]).astype(np.float32)
#HomoMat =cv2.getPerspectiveTransform(roadPts4Homography[:,::1], roadDesiredPts[:,::1])
HomoMat = None

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
	width, height, rgbImg, depthImg, segImg= p.getCameraImage(640,480,viewMatrix=viewMat,projectionMatrix=projMat, renderer=p.ER_BULLET_HARDWARE_OPENGL)
	# faccio un reshape in quanto da sopra ottengo un array di elementi
	

	# Tolgo il canale alpha e converto da BGRA a RGB per la rete
	imgHSV = cv2.cvtColor(rgbImg[:,:,:3], cv2.COLOR_RGB2HSV)
	skyMsk = cv2.inRange(imgHSV[:,:,0], 115,125)
	greenMsk = cv2.inRange(imgHSV[:,:,0], 55,65)
 

 
	greenMsk = cv2.morphologyEx(greenMsk, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(1,3)), iterations=2)
	skyMsk = cv2.morphologyEx(skyMsk, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(1,3)), iterations=4)
	
	findEdge = lambda x, line: np.argwhere(np.abs(np.diff(x[line,:].astype(np.int16)))==255)
	topEdge = findEdge(greenMsk,320).flatten()
	botEdge = findEdge(greenMsk, 460).flatten()
 
#	print(f'BOT{botEdge}')
#	print(f'TOP{topEdge}')
	# global HomoMat
	# if HomoMat is None and len(topEdge)==2 and len(botEdge)==2:
	# 	origPts = np.array([[topEdge[0], 320],[topEdge[1], 320],[botEdge[1], 460],[botEdge[0], 460]]).astype(np.float32)
	# 	destPts = np.array([[250,360],[390,360],[390,460],[250,460]]).astype(np.float32)
	# 	HomoMat =cv2.getPerspectiveTransform(origPts, destPts)

	HomoMat = np.array([[-2.98962782e-01, -1.24344112e+00,  3.93075046e+02],
	[-3.38241831e-15, -2.16351434e+00,  5.70860281e+02],
	[-8.01126080e-18, -4.17937767e-03,  1.00000000e+00]])
	
	#print(colorOnly.shape)
	cv2.imshow('testIMAGE', imgHSV)

	cv2.imshow('rectified', cv2.warpPerspective(255-(greenMsk+skyMsk), HomoMat, (640,480),flags=cv2.INTER_LINEAR)[::3,::3])
	cv2.waitKey(1)
	#print("Reshaped RGB image size:", rgb_image.shape)

	return greenMsk

					
def birdEyeView(image):

	# Setting parameter values 
	t_lower = 120  # Lower Threshold 
	t_upper = 210  # Upper threshold 
	
	# immagine canny
	#edge = cv2.Canny(image, t_lower, t_upper) 
	#y,x = np.where(edge>0)

	#image_edges = image.copy()
	
	
	
	#edges = cv2.cvtColor(edge,cv2.COLOR_GRAY2RGB)
	#edges[y,x] = [255,255,255]

	#cv2.imshow("Camera", edges)
	#cv2.waitKey(0) 

	#abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0))
	#abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1))
	#scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	#print(scaled_sobel)
	#edge = cv2.Canny(img, 300, 355,L2gradient=True)
	
	#mask = cv2.inRange(image, t_lower, t_upper)
	#lab_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

	# Usate per rimuovere il rumore
	# lower = np.array([0, 140, 140])
	# upper = np.array([255, 190, 190])
	#mask = cv2.inRange(lab_img, lower, upper)
	#filter1 = cv2.morphologyEx(abs_sobel, cv2.MORPH_OPEN, np.ones((1, 5), np.uint8), iterations=1)
	#filter = cv2.morphologyEx(filter1, cv2.MORPH_CLOSE, np.ones((5, 1), np.uint8), iterations=1)
	
	#ret, filtered = cv2.threshold(edges, 5, 255, cv2.THRESH_BINARY)
	# blur = cv2.GaussianBlur(edge, (0,0), sigmaX=33, sigmaY=33)

	# # # divide
	# divide = cv2.divide(edge, blur, scale=255)

	# # # otsu threshold
	# thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

	# # # apply morphology
	# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
	# morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

	#img = cv2.undistort(image,mtx,dist,None,mtx)
	img_size = (image.shape[1],image.shape[0])   
	

	# # Combina i gradienti in uscita dalla funzione di Sobel con la binary image
	#preprocessImage = np.zeros_like(img[:,:,0]) # canale blue
	# gradx = self.abs_sobel_thresh(img, orient='x', thresh_min=12, thresh_max=255)
	# grady = self.abs_sobel_thresh(img, orient='y', thresh_min=25, thresh_max=255)
	# c_binary = self.color_threshold(img, sthresh=(100,255), vthresh=(50,255))
	# Questa parte seleziona i pixel in preprocessImage dove sia gradx che grady sono uguali a 1 oppure c_binary 
	# è uguale a 1.
	# = 255: infine imposta il valore di intensità di questi pixel selezionati su 255 ovvero rendendoli bianchi.
	#preprocessImage[((gradx == 1) & (grady ==1) | (c_binary == 1))] = 255

	# scelta dei 4 punti nell'immagine di partenza da mappare poi nei punti di destinazione dst dell'immagine finale
	# questi 4 punti sono la regione d'interesse
	# basso sinistra, alto sinistra, alto destra, basso destra
	#src = np.float32([[img.shape[1]*(0.5-self.mid_width/2), img.shape[0]*self.height_pct],[img.shape[1]*(0.5+self.mid_width/2),img.shape[0]*self.height_pct],[img.shape[1]*(0.5+self.bot_width/2), img.shape[0]*self.bottom_trim],[img.shape[1]*(0.5-self.bot_width/2), img.shape[0]*self.bottom_trim]])
	#print(img_size)
	#src = np.float32([[25,44],[188,44],[188,65],[25,65]])
	src = np.float32([[235,368],[465,368],[580,479],[155,479]])
	
	# for i in src:
	#         print(i)
	#         cv2.circle(image,[int(i[0]),int(i[1])],2,(0,0,255),-1)
			
	
	dst = np.float32([[0+150,0],[img_size[0]-150,0],[img_size[0]-150,img_size[1]],[0+150,img_size[1]]])


	#print(src)
	# con la funzione cv2.getPerspectiveTransform otteniamo la matrice di rotazione che ci porta dai punti
	# sorgenti a quelli di destinazione
	M = cv2.getPerspectiveTransform(src,dst)

	# la funzione cv2.warpPerspective viene utilizzata per applicare la Perspective Transform ad un'immagine,
	# data la matrice di trasformazione
	# INPUT:
	# input image
	# transformation matrix
	# dimensioni immagine finale
	# metodo di interpolazione
	bird_eye = cv2.warpPerspective(edge,M,(640,480),flags=cv2.INTER_LINEAR)
	


	# viene ritornata anche questa in quanto permette di avere la birdEye view sull'immagine originale
	#warped_original = cv2.warpPerspective(img,M,img_size,flags=cv2.INTER_LINEAR)

	return edge


def wrap2pi(angle):
	return (angle + np.pi) % (2 * np.pi) - np.pi


def rect_to_polar_relative(goal):

	"""
	Funzione usata per la trasformazione in coordinate polari del goal
	
	Parameters:
	- robot_id: id del robot
	
	Returns:
	- r: raggio, ovvero la distanza tra robot e goal
	- theta: angolo di orientazione del robot rispetto al goal
	"""
	
	# posizioe del goal
	goal = goal
	
	# calcolo la posizione correte del robot specificato
	robPos, car_orientation = p.getBasePositionAndOrientation(turtle)
	#print("pos: ",robPos)
	# calcolo l'angolo di yaw
	_,_,yaw = p.getEulerFromQuaternion(car_orientation)
	
	# Calculate the polar coordinates (distance, angle) of the vector
	vector_to_goal = np.array([goal[0] - robPos[0], goal[1] - robPos[1]])
	r = np.linalg.norm(vector_to_goal)
	theta = wrap2pi(np.arctan2(vector_to_goal[1], vector_to_goal[0])-wrap2pi(yaw))
	return r, theta


def p_control(yaw_error):
	kp = 0.9
	output = kp*yaw_error
	return output


def choosePositionAndIndex(position,index):
	if len(position!=0):
		for i,j in zip(range(len(position)),position):
			r, yaw_error = rect_to_polar_relative(j[:2])
			vel_ang = p_control(yaw_error)
			print(f"steer {vel_ang} - distance {r}")
			if vel_ang < 1.5:
				positionToStart = j[:2]
				indexToStart = index[i]
				done = True
			else:
				positionToStart = []
				indexToStart = None
				done = False
	else:
		positionToStart = []
		indexToStart = None
		done = False
	return positionToStart, indexToStart, done




model = model_new.VehicleControlModel()

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetDebugVisualizerCamera(cameraDistance=22, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[4,-2,0])
#p.resetDebugVisualizerCamera(cameraDistance=22, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[0,0,0])

#p.loadURDF("plane.urdf")

#turtle = p.loadURDF("f10_racecar/simplecar.urdf", [-10,1,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(180)]))
#turtle = p.loadURDF("f10_racecar/simplecar.urdf", [-12,-11,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(180)]))
#turtle = p.loadURDF("f10_racecar/simplecar.urdf", [-9,-6.5,.3])
#turtle = p.loadURDF("f10_racecar/simplecar.urdf", [0,0,.3],  p.getQuaternionFromEuler([0,0,np.deg2rad(55)]))
turtle = p.loadURDF("f10_racecar/simplecar.urdf", [35.5,2,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(-90)]))
track = p.loadSDF("f10_racecar/meshes/barca_track_modified.sdf", globalScaling=1)


env1 = [[-10,1,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(180)])]
env2 = [[-12,-11,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(180)])]
env3 = [[-9,-6.5,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(0)])]
env4 = [[0,0,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(55)])]
env5 = [[35,2,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(-90)])]

env_list = [env1,env2,env3,env4,env5]
index_env = random.randint(0,len(env_list)-1)
print(index_env)
#turtle = p.loadURDF("f10_racecar/simplecar.urdf", env_list[index_env][0],env_list[index_env][1])
#turtle = p.loadURDF("f10_racecar/simplecar.urdf", [-12,2,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(-90)]))



robPos, car_orientation = p.getBasePositionAndOrientation(turtle)

# calcolo l'angolo di yaw
_,_,yaw = p.getEulerFromQuaternion(car_orientation)
#print("yaw: ",yaw)
	   
#turtle = p.loadURDF("f10_racecar/simplecar.urdf", [0,0,.3])
#turtle = p.loadURDF("f10_racecar/simplecar.urdf", [0,0,.3])
# sphere_color = [1, 0, 0,1]  # Red color
# sphere_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=0)
# sphere_body = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=sphere_shape,basePosition=[-10,5, 0])
# p.changeVisualShape(sphere_body, -1, rgbaColor=sphere_color)
#track = p.loadSDF("f10_racecar/meshes/barca_track.sdf", globalScaling=1)
p.setRealTimeSimulation(1)


# image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
# image = np.transpose(image, (2, 0, 1))
#print(image.shape)
#print(image[np.newaxis,...].shape)
#prediction = model(torch.from_numpy(image[np.newaxis,...]).type(torch.FloatTensor))

#print(prediction)
#p.setGravity(0,0,-9.81)
# sphere_color = [1, 0, 0,1]  # Red color
# sphere_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=0)
# sphere_body = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=sphere_shape,basePosition=[0,1,0 ])
# p.changeVisualShape(sphere_body, -1, rgbaColor=sphere_color)



# for j in range (p.getNumJoints(turtle)):
# 	print(p.getJointInfo(turtle,j))
forward=0
turn=0
while (1):
	p.setGravity(0,0,-9.81)

	# time.sleep(1./240.)
	# keys = p.getKeyboardEvents()
	# leftWheelVelocity=0
	# rightWheelVelocity=0
	# speed=10
	
	# for k,v in keys.items():

	#         if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
	#                 turn = -0.5
	#         if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_RELEASED)):
	#                 turn = 0
	#         if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
	#                 turn = 0.5
	#         if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_RELEASED)):
	#                 turn = 0

	#         if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_TRIGGERED)):
	#                 forward=vel_max
	#         if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_RELEASED)):
	#                 forward=0
	#         if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_TRIGGERED)):
	#                 forward=-vel_max
	#         if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_RELEASED)):
	#                 forward=0

	
	car_position, car_orientation = p.getBasePositionAndOrientation(turtle)
	
	_,_,centerLine = computeTrack(debug=False)
	
	positionToStart = []
	while len(positionToStart)==0:
		index,position = getPointToStart(centerLine,car_position[:2], threshold)
		positionToStart, indexToStart, done = choosePositionAndIndex(position,index)
		print(position)
		threshold+=0.05
			
	
	#print("Pos: ",positionToStart)
	
	
			
	# creo le liste di indici 
	indexList_1 = [idx for idx in range(indexToStart,len(centerLine),1)]
	indexList_2 = [idx for idx in range(0,indexToStart,1)]
	
	total_index = indexList_1+indexList_2
	#print(indexList_2)
	
	
	for i in total_index:
		#print("Index: ",i)
		goal = centerLine[i][:2] # non prendo la z
		#print("Goal: ",goal)
		r, yaw_error = rect_to_polar_relative(goal)
		print(f"goal {goal},distance {r}")
		#print("cambio")
		
		while r>0.5:
			img = getCamera_image()
			
			#bird_eye = birdEyeView(img)


			reshaped_image = cv2.resize(img, (96, 84))
			#print(reshaped_image.shape)
			#no_greeen = cv2.inRange(img,np.array([136,169,60]),np.array([141,171,60]))
			#no_greeen = cv2.inRange(img,np.array([80,8,120]),np.array([90,5,150]))
			#cv2.imshow("Camera", no_greeen)
			
   			
			
			#cv2.waitKey(1)

			r, yaw_error = rect_to_polar_relative(goal)
			#print(f"goal {goal},distance {r} ,yaw_error{yaw_error}")
			vel_ang = p_control(yaw_error)
			#forward = p_control()
			#print(f"steer {vel_ang} - distance {r}")

			forward = 10
			# if 154<i<164:
			#         forward = 4
			# if 199<i<210:
			# 	print("vel_ang; ",vel_ang)
			# 	print(yaw_error)
			# 	forward = 2
			
			#print("Velocità: ",forward)
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
			p.setJointMotorControl2(turtle,0,p.POSITION_CONTROL,targetPosition=vel_ang)
			p.setJointMotorControl2(turtle,2,p.POSITION_CONTROL,targetPosition=vel_ang)
			p.setJointMotorControl2(turtle,1,p.VELOCITY_CONTROL,targetVelocity=forward)
			p.setJointMotorControl2(turtle,3,p.VELOCITY_CONTROL,targetVelocity=forward)
			p.setJointMotorControl2(turtle,4,p.VELOCITY_CONTROL,targetVelocity=forward)
			p.setJointMotorControl2(turtle,5,p.VELOCITY_CONTROL,targetVelocity=forward)




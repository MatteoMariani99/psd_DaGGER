import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import cv2
from itertools import chain
from get_track import *


zed_camera_joint = 7 # simplecar
# zed_camera_joint = 5 racecar



class PyBulletContinuousEnv(gym.Env):
    def __init__(self, total_episode_step=1000):
        super(PyBulletContinuousEnv, self).__init__()

        # Connect to PyBullet and set up the environment
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setRealTimeSimulation(1)
        
        #self.threshold = 0.25 # soglia usata per trovare i punti più vicini alla posizione del robot

        
        # Define action and observation space
        # Continuous action space: steer (left/right), up, down compresi tra -1, +1
        #self.action_space = spaces.Box( np.array([-1,0,0]), np.array([+1,+1,+1]), dtype=np.float32)  # steer, gas, brake
        #self.action_space = spaces.Box( np.array([-1,-1]), np.array([+1,+1]), dtype=np.float32)  # steer, gas/brake


        # Sono le immagini di dimensioni 200x66
        #self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_W, STATE_H, 3), dtype=np.uint8)

        self.total_episode_steps = total_episode_step

    def reset(self):
        self.current_steps = 0
        # Reset the simulation and the cartpole position
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.resetDebugVisualizerCamera(cameraDistance=20, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[-4,-2,0])
        #p.loadURDF("plane.urdf",useFixedBase = True)

        # for i in range(0,30,1):
        #     p.loadURDF("cube/marble_cube.urdf",[i,-1.5,0],useFixedBase = True)
        # for l in range(0,30,1):
        #     p.loadURDF("cube/marble_cube.urdf",[l,1.5,0],useFixedBase = True)

        # p.loadURDF("cube/marble_cube.urdf",[29,0,0],useFixedBase = True)
        p.loadSDF("f10_racecar/meshes/barca_track_modified.sdf", globalScaling=1)
        

        self.car_id = p.loadURDF("f10_racecar/simplecar.urdf", [-9,-6.5,.3])
        #print()
        #print(p.getEulerFromQuaternion(p.getQuaternionFromEuler([0,0,100]))[2]*180/(2*3.14))
        # sphere_color = [1, 0, 0,1]  # Red color
        # sphere_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=0)
        # sphere_body = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=sphere_shape,basePosition=[-10,8, 0])
        # p.changeVisualShape(sphere_body, -1, rgbaColor=sphere_color)


    # Le osservazioni sono le immagini 96x96 rgb prese dalla zed
    def get_observation(self):
        
        #velocity_vector = []
        #pose_vector = []

        # Nel caso in cui servano posizione e velocità rispetto al frame world
        #car_position, car_orientation = p.getBasePositionAndOrientation(self.car_id)
        #car_velocity_x, car_angular_velocity_z = self.getVelocity()

        #_,_,yaw = p.getEulerFromQuaternion(car_orientation)


        #velocity_vector.append([car_velocity_x, car_angular_velocity_z])
        #pose_vector.append([car_position[0], car_position[1],yaw])

        rgb_image = self.getCamera_image()
        bird_eye_state = self.birdEyeView(rgb_image)
        
        # immagine in formato bird-eye -> canny filter
        
        return bird_eye_state


    def step(self, action):
        self.action = action
        #print("Azione: ",action)
        print("Step: ", self.current_steps)
        self.current_steps += 1
        done = False

        steer = action[0]
        forward = action[1]
        #backward = action[2]
        
        # SIMPLECAR
        # ctrl+shift+l per fare il replace di una variabile
        # ruote anteriori
        p.setJointMotorControl2(self.car_id,1,p.VELOCITY_CONTROL,targetVelocity=forward)
        p.setJointMotorControl2(self.car_id,3,p.VELOCITY_CONTROL,targetVelocity=forward)
        # ruote posteriori
        p.setJointMotorControl2(self.car_id,4,p.VELOCITY_CONTROL,targetVelocity=forward)
        p.setJointMotorControl2(self.car_id,5,p.VELOCITY_CONTROL,targetVelocity=forward)
        # sterzo
        p.setJointMotorControl2(self.car_id,0,p.POSITION_CONTROL,targetPosition=steer)
        p.setJointMotorControl2(self.car_id,2,p.POSITION_CONTROL,targetPosition=steer)


        # F10 RACECAR
        # ruote anteriori
        # p.setJointMotorControl2(self.car_id,1,p.VELOCITY_CONTROL,targetVelocity=forward-backward)
        # p.setJointMotorControl2(self.car_id,3,p.VELOCITY_CONTROL,targetVelocity=forward-backward)
        # # ruote posteriori
        # p.setJointMotorControl2(self.car_id,4,p.VELOCITY_CONTROL,targetVelocity=forward-backward)
        # p.setJointMotorControl2(self.car_id,5,p.VELOCITY_CONTROL,targetVelocity=forward-backward)
        # # sterzo
        # p.setJointMotorControl2(self.car_id,0,p.POSITION_CONTROL,targetPosition=-steer)
        # p.setJointMotorControl2(self.car_id,2,p.POSITION_CONTROL,targetPosition=-steer)

        # Step di simulazione
        #for _ in range(240):
        #    p.stepSimulation()

        # Ottengo lo stato che sarebbero le ossservazioni (immagine 96x96x3)
        state = self.get_observation()

        reward = 1 # per ora lascio 0
        
        # quando il numero di step super quello degli episodi, done diventa true e si conclude l'episodio
        if self.current_steps > self.total_episode_steps:
            done = True

        return state, reward, done


    # chiudo tutto
    def close(self):
        p.disconnect()


    # ottengo l'immagine dalla zed montata sul robot e ritorno l'immagine rgb 96x96x3
    def getCamera_image(self):
        camInfo = p.getDebugVisualizerCamera()
        ls = p.getLinkState(self.car_id,zed_camera_joint, computeForwardKinematics=True)
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
        
        width, height, rgbImg, depthImg, segImg= p.getCameraImage(96,96,viewMatrix=viewMat,projectionMatrix=projMat, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        # faccio un reshape in quanto da sopra ottengo un array di elementi
        rgb_opengl = np.reshape(rgbImg, (96, 96, 4)) 

        # Tolgo il canale alpha e converto da BGRA a RGB per la rete
        rgb_image = cv2.cvtColor(rgb_opengl, cv2.COLOR_BGRA2RGB)
        #print("Reshaped RGB image size:", rgb_image.shape)

        return rgb_image


    # velocità rispetto al frame macchina
    def getVelocity(self):
        linearVelocity, angularVelocity = p.getBaseVelocity(self.car_id)
        base_pose =  p.getBasePositionAndOrientation(self.car_id)

        # Converto le velocità angolari e lineari in local frame
        rotation_matrix = p.getMatrixFromQuaternion(base_pose[1])  # Rotation matrix from quaternion
        rotation_matrix = np.reshape(rotation_matrix, (3, 3))

        # Velocità lineare nel frame locale
        local_linear_velocity = np.dot(np.linalg.inv(rotation_matrix), np.array(linearVelocity))

        # Velocità angolare nel frame locale
        local_angular_velocity = np.dot(np.linalg.inv(rotation_matrix), np.array(angularVelocity))
        
        # Extract the velocity values
        linear_velocity = np.array(local_linear_velocity)[0] # x velocity
        angular_velocity = np.array(local_angular_velocity)[2] # z angular

        return linear_velocity, angular_velocity
    

    # visualizzazione a video dell'immagine corrente
    def visualization_image(self):
        camInfo = p.getDebugVisualizerCamera()
        ls = p.getLinkState(self.car_id,zed_camera_joint, computeForwardKinematics=True)
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
        rgb_opengl = np.reshape(rgbImg, (480, 640, 4)) 
        

        # Tolgo il canale alpha e converto da BGRA a RGB per la rete
        rgb_image = cv2.cvtColor(rgb_opengl, cv2.COLOR_BGRA2RGB)

        # font 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        
        # org (w,h)
        org = (30, 30) 
        org1 = (30,65)
        
        # fontScale 
        fontScale = 1
        
        # Blue color in BGR 
        color = (0, 255, 0) 
        color1 = (0,0,255)
        
        # Line thickness of 3 px 
        thickness = 3

        # Testo
        text = "Ster: "
        full_text = f"{text}{self.action[0]}" 
        text = "Gas: "
        full_text1 = f"{text}{self.action[1]}" 
        
        # Display del testo a video
        image = cv2.putText(rgb_image, full_text, org, font,  
                        fontScale, color, thickness, cv2.LINE_AA) 
        image = cv2.putText(image, full_text1, org1, font,  
                        fontScale, color1, thickness, cv2.LINE_AA) 

        return image


    def stoppingCar(self):
        steer = 0
        forward = 0
        #backward = 0
        
        # ctrl+shift+l per fare il replace di una variabile
        # ruote anteriori
        p.setJointMotorControl2(self.car_id,1,p.VELOCITY_CONTROL,targetVelocity=forward)
        p.setJointMotorControl2(self.car_id,3,p.VELOCITY_CONTROL,targetVelocity=forward)
        # ruote posteriori
        p.setJointMotorControl2(self.car_id,4,p.VELOCITY_CONTROL,targetVelocity=forward)
        p.setJointMotorControl2(self.car_id,5,p.VELOCITY_CONTROL,targetVelocity=forward)
        # sterzo
        p.setJointMotorControl2(self.car_id,0,p.POSITION_CONTROL,targetPosition=-steer)
        p.setJointMotorControl2(self.car_id,2,p.POSITION_CONTROL,targetPosition=-steer)

        print("Car stopped!")



    def getCarPosition(self):
        car_position, car_orientation = p.getBasePositionAndOrientation(self.car_id)
        
        return car_position, car_orientation
    
    
    def wrap2pi(self,angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi


    def rect_to_polar_relative(self,goal):
        """
        Funzione usata per la trasformazione in coordinate polari del goal
        
        Parameters:
        - robot_id: id del robot
        
        Returns:
        - r: raggio, ovvero la distanza tra robot e goal
        - theta: angolo di orientazione del robot rispetto al goal
        """
        #print("G; ",goal)
        # calcolo la posizione correte del robot specificato
        car_position, car_orientation = self.getCarPosition()
        #print("Posizione: ",car_position)
        # calcolo l'angolo di yaw
        _,_,yaw = p.getEulerFromQuaternion(car_orientation)
        
        # Calculate the polar coordinates (distance, angle) of the vector
        vector_to_goal = np.array([goal[0] - car_position[0], goal[1] - car_position[1]])
        r = np.linalg.norm(vector_to_goal)
        theta = self.wrap2pi(np.arctan2(vector_to_goal[1], vector_to_goal[0])-self.wrap2pi(yaw))
        return r, theta


    def choosePositionAndIndex(self,position,index):
        if len(position!=0):
            for i,j in zip(range(len(position)),position):
                r, yaw_error = self.rect_to_polar_relative(j[:2])
                vel_ang = self.p_control(yaw_error)
                #print(f"steer {vel_ang} - distance {r}")
                if vel_ang < 1.5:
                    positionToStart = j[:2]
                    indexToStart = index[i]
                    #done = True
                else:
                    positionToStart = []
                    indexToStart = None
                    #done = False
        else:
            positionToStart = []
            indexToStart = None
            #done = False
        return positionToStart, indexToStart
    
    
    def p_control(self,yaw_error):
        kp = 0.9
        output = kp*yaw_error
        return output
    

    def birdEyeView(self,image):

        # Setting parameter values 
        t_lower = 330  # Lower Threshold 
        t_upper = 350  # Upper threshold 
        
        # immagine canny
        #edge = cv2.Canny(image, t_lower, t_upper)
        abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1))
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        edge = cv2.Canny(scaled_sobel, t_lower, t_upper)
        
        # # blur
        #blur = cv2.GaussianBlur(edge, (0,0), sigmaX=33, sigmaY=33)

        # # divide
        # divide = cv2.divide(edge, blur, scale=255)

        # # otsu threshold
        # thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

        # # apply morphology
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        # morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        
        # Per colorare gli edge 
        #y,x = np.where(edge>0)
        #edges = cv2.cvtColor(edge,cv2.COLOR_GRAY2RGB)
        #edges[y,x] = [255,0,0]
  

        #filter1 = cv2.morphologyEx(edge, cv2.MORPH_OPEN, np.ones((1, 3), np.uint8), iterations=1)
        #filter = cv2.morphologyEx(filter1, cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8), iterations=1)
        #ret, filtered = cv2.threshold(edges, 5, 255, cv2.THRESH_BINARY)
        
        #img = cv2.undistort(image,mtx,dist,None,mtx)
        img_size = (image.shape[1],image.shape[0])   

        # punti sorgenti da mappare in punti destinazione
        src = np.float32([[235,368],[465,368],[580,479],[155,479]])
        dst = np.float32([[0+150,0],[img_size[0]-150,0],[img_size[0]-150,img_size[1]],[0+150,img_size[1]]])

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
        reshaped_image = cv2.resize(edge, (96, 84))
        #cv2.imshow("Camera", reshaped_image)
        #cv2.waitKey(1) 
   
        return reshaped_image


    def computeIndexToStart(self, centerLine):
        positionToStart = []
        threshold = 0.25
        car_position, _ = self.getCarPosition()
    
        
        # lista delle possibili posizioni di partenza
        while len(positionToStart)==0:
            index,position = getPointToStart(centerLine,car_position[:2], threshold)
            positionToStart, indexToStart = self.choosePositionAndIndex(position,index)
            threshold+=0.05
                
        #print("Pos: ",positionToStart)
        
        # creo le liste di indici 
        indexList_1 = [idx for idx in range(indexToStart,len(centerLine),1)]
        indexList_2 = [idx for idx in range(0,indexToStart,1)]
        
        total_index = indexList_1+indexList_2
        
        return total_index
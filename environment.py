import gym
import numpy as np
import pybullet as p
import pybullet_data
import cv2
from get_track import *
import random


zed_camera_joint = 7 # simplecar



class PyBulletContinuousEnv(gym.Env):
    def __init__(self, total_episode_step=2000):
        super(PyBulletContinuousEnv, self).__init__()

        # Connessione a PyBullet e setup della simulazione
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setRealTimeSimulation(1)

        # numero totale di episodi
        self.total_episode_steps = total_episode_step
        # matrice per l'omografia
        self.HomoMat = None


    # funzione utile a inizializzare l'ambiente 
    def reset(self):
        self.current_steps = 0
        
        # Reset della simulazione
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.resetDebugVisualizerCamera(cameraDistance=20, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[4,-2,0])
 
        # carico il tracciato
        p.loadSDF("world&car/meshes/barca_track_modified.sdf", globalScaling=1)
        
        # punti si spawn della macchina
        env1 = [[-10,1,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(180)])]
        env2 = [[-12,-11,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(180)])]
        env3 = [[-9,-6.5,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(0)])]
        env4 = [[0,0,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(55)])]
        env5 = [[35.5,2,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(-90)])]

        env_list = [env1,env2,env3,env4,env5]
        index_env = random.randint(0,len(env_list)-1)
        self.car_id = p.loadURDF("world&car/simplecar.urdf", env_list[index_env][0],env_list[index_env][1])


    # Le osservazioni sono le immagini 84x96 rgb prese dalla zed
    def get_observation(self):

        rgb_image = self.getCamera_image()
        reshaped_bird_eye = self.birdEyeView(rgb_image)
        
        return reshaped_bird_eye


    # funzione step che definisce il passo di simulazione
    def step(self, action):
        self.action = action
        
        print("Step: ", self.current_steps)
        self.current_steps += 1
        done = False

        # il primo elemento delle azioni è lo sterzo, mentre il secondo è la velocità
        steer = action[0]
        forward = action[1]
        
        
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

        # per andare a 10Hz
        # for _ in range(24):
        #     p.stepSimulation()

        # Ottengo lo stato che sarebbero le ossservazioni (immagine 84x96x3)
        state = self.get_observation()

        reward = 1 # per ora lascio 1
        
        # quando il numero di step super quello degli episodi, done diventa true e si conclude l'episodio
        if self.current_steps > self.total_episode_steps:
            done = True

        return state, reward, done


    # chiudo tutto
    def close(self):
        p.disconnect()


    # ottengo l'immagine dalla zed montata sul robot e ritorno l'immagine HSV 640X480x3
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
        
        width, height, rgbImg, depthImg, segImg= p.getCameraImage(640,480,viewMatrix=viewMat,projectionMatrix=projMat, renderer=p.ER_BULLET_HARDWARE_OPENGL)

        imgHSV = cv2.cvtColor(rgbImg[:,:,:3], cv2.COLOR_RGB2HSV)
        return imgHSV


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
    

    # fermo la macchina alla fine della simulazione prima del training
    def stoppingCar(self):
        steer = 0
        forward = 0
        
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


    # funzione che restituisce la posizione e orientazione della macchina
    def getCarPosition(self):
        car_position, car_orientation = p.getBasePositionAndOrientation(self.car_id)
        
        return car_position, car_orientation
    
    
    # mappa l'angolo tra +- pi
    def wrap2pi(self,angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi


    # funzione che mi restituisce le coordinate polari del goal (punto della pista) che devo raggiungere con l'expert
    def rect_to_polar_relative(self,goal):
        """
        Funzione usata per la trasformazione in coordinate polari del goal
        
        Parameters:
        - robot_id: id del robot
        
        Returns:
        - r: raggio, ovvero la distanza tra robot e goal
        - theta: angolo di orientazione del robot rispetto al goal
        """
        
        # calcolo la posizione correte del robot specificato
        car_position, car_orientation = self.getCarPosition()
        
        # calcolo l'angolo di yaw
        _,_,yaw = p.getEulerFromQuaternion(car_orientation)
        
        # Calculate the polar coordinates (distance, angle) of the vector
        vector_to_goal = np.array([goal[0] - car_position[0], goal[1] - car_position[1]])
        r = np.linalg.norm(vector_to_goal)
        theta = self.wrap2pi(np.arctan2(vector_to_goal[1], vector_to_goal[0])-self.wrap2pi(yaw))
        return r, theta


    # funzione che restituisce la posizione da cui partire in termini di coordinate e il corrispondente indice della lista
    def choosePositionAndIndex(self,position,index):
        if len(position!=0):
            for i,j in zip(range(len(position)),position):
                r, yaw_error = self.rect_to_polar_relative(j[:2])
                vel_ang,_ = self.p_control(yaw_error)
                
                if vel_ang < 1.5:
                    positionToStart = j[:2]
                    indexToStart = index[i]
                    
                else:
                    positionToStart = []
                    indexToStart = None
                    
        else:
            positionToStart = []
            indexToStart = None
            
        return positionToStart, indexToStart
    
    
    # funzione relativa al controllore P: in uscita ottengo sterzo (proporzionale all'errore sulla yaw) e velocità dell'expert
    def p_control(self,yaw_error):
        kp = 0.9
        vel_ang = kp*yaw_error
        vel_lin = 10 # m/s
    
        if abs(yaw_error)>0.1:
            vel_lin = (2.3-abs(vel_ang))*(10/2.3)
        return vel_ang, vel_lin
    

    # funzione che mi permette di ottenere la birdEye e restituisce in uscita l'immagine 84x96x1
    def birdEyeView(self,imgHSV):

        # la funzione cv2.warpPerspective viene utilizzata per applicare la Perspective Transform ad un'immagine,
        # data la matrice di trasformazione
        # INPUT:
        # input image
        # transformation matrix
        # dimensioni immagine finale
        # metodo di interpolazione
        
        skyMsk = cv2.inRange(imgHSV[:,:,0], 115,125) # maschera del cielo
        greenMsk = cv2.inRange(imgHSV[:,:,0], 55,65) # maschera del prato
    
        # rimuovo il rumore 
        greenMsk = cv2.morphologyEx(greenMsk, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(1,3)), iterations=2)
        skyMsk = cv2.morphologyEx(skyMsk, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(1,3)), iterations=4)
        
        # trovo in automatico i punti sorgenti per la bird eye: dove i pixel sono neri -> 255
        #findEdge = lambda x, line: np.argwhere(np.abs(np.diff(x[line,:].astype(np.int16)))==255)
        #topEdge = findEdge(greenMsk,320).flatten()
        #botEdge = findEdge(greenMsk, 460).flatten()

        # calcolo l'omografia una sola volta
        # if self.HomoMat is None and len(topEdge)==2 and len(botEdge)==2:
        #     origPts = np.array([[topEdge[0], 320],[topEdge[1], 320],[botEdge[1], 460],[botEdge[0], 460]]).astype(np.float32)
        #     destPts = np.array([[250,360],[390,360],[390,460],[250,460]]).astype(np.float32)
        #     self.HomoMat =cv2.getPerspectiveTransform(origPts, destPts)
        
        # HomoMat calcolata sola la prima volta
        self.HomoMat = np.array([[-2.98962782e-01, -1.24344112e+00,  3.93075046e+02],
            [-3.38241831e-15, -2.16351434e+00,  5.70860281e+02],
            [-8.01126080e-18, -4.17937767e-03,  1.00000000e+00]])
    
        # birdeye togliendo il cielo e il prato
        # risultato: cielo e prato neri e la strada bianca
        bird_eye = cv2.warpPerspective(255-(greenMsk+skyMsk), self.HomoMat, (640,480),flags=cv2.INTER_LINEAR)[::3,::3]
        
        reshaped_image = cv2.resize(bird_eye, (96, 84)) # larghezza, altezza
        
        #cv2.imshow('testIMAGE', imgHSV)
        #cv2.imshow('rectified', bird_eye )
        #cv2.imshow("Camera", reshaped_image)
        #cv2.waitKey(1) 
   
        return reshaped_image


    # funzione che mi restituisce la lista completa e riordinata della linea centrale: l'inzio è la posizione di partenza 
    # della macchina
    def computeIndexToStart(self, centerLine):
        positionToStart = []
        threshold = 0.25
        car_position, _ = self.getCarPosition()
    
        # lista delle possibili posizioni di partenza
        while len(positionToStart)==0:
            index,position = getPointToStart(centerLine,car_position[:2], threshold)
            positionToStart, indexToStart = self.choosePositionAndIndex(position,index)
            threshold+=0.05
                
        
        # creo le liste di indici 
        indexList_1 = [idx for idx in range(indexToStart,len(centerLine),1)]
        indexList_2 = [idx for idx in range(0,indexToStart,1)]
        
        total_index = indexList_1+indexList_2
        
        return total_index
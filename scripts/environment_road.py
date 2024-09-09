import gym
import numpy as np
import pybullet as p
import pybullet_data
import cv2
from get_track import *
import random

# giunto relativo alla camera sul veicolo "simplecar"
zed_camera_joint = 7 # simplecar



class RoadEnv(gym.Env):
    def __init__(self, total_episode_step=1999):
        super(RoadEnv, self).__init__()

        # Connessione a PyBullet e setup della simulazione
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.total_episode_steps = total_episode_step # numero totale di episodi
        self.HomoMat = None # matrice per l'omografia


 
    def reset(self):
        """
        Funzione che permette di inizializzare l'ambiente (tracciato e veicolo)
        """
        self.current_steps = 0
        
        # Reset della simulazione
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.resetDebugVisualizerCamera(cameraDistance=20, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[4,-2,0])
 
        # caricamento del tracciato
        p.loadSDF("world/models/road/meshes/barca_track_modified.sdf", globalScaling=1)

        # punti di spawn della macchina
        train = False
        if train:
            env1 = [[-10,1,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(180)])]
            env2 = [[-12,-11,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(180)])]
            env3 = [[-9,-6.5,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(0)])]
            env4 = [[0,0,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(55)])]
            env5 = [[35.5,2,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(-90)])]
            env_list = [env1, env2, env3, env4, env5]
        else:
            env5 = [[35.5,2,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(-90)])]
            env6 = [[27.6,-4.1,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(90)])] # reverse for testing
            env_list = [env5, env6]
            
        index_env = random.randint(0,len(env_list)-1)
        self.car_id = p.loadURDF("world/models/car/simplecar.urdf", env_list[index_env][0],env_list[index_env][1])
        
    

    def get_observation(self):
        """
        Funzione che restituisce le osservazioni, ovvero immagini 84x96x1 birdEye, utili all'algoritmo.
        
        Returns:
            - reshaped_bird_eye: immagini birdEye con resize a 84x96x1
        """
        hsv_image = self.getCamera_image()
        
        # funzione che mi definisce la bird eye
        reshaped_bird_eye = self.birdEyeView(hsv_image)
        
        return reshaped_bird_eye



    def step(self, action):
        """
        Funzione usata per eseguire il passo di simulazione una volta calcolate le azioni.
        
        Parameters:
        - action: rappresenta la velocità angolare (action[0]) e la velocità lineare (action[1]) da comandare al veicolo
        
        Returns:
        - state: immagini 84x96x1 birdEye
        - reward: (non utilizzata)
        - done: flag che rappresenta l'azone terminale
        """
        self.action = action
        
        print("Step: ", self.current_steps)
        self.current_steps += 1
        done = False

        steer = action[0] # comando di sterzo
        forward = action[1] # comando di velocità lineare
        
        
        # Setting delle velocità ai rispettivi giunti della SIMPLECAR
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
        for _ in range(24):
            p.stepSimulation()

        # Ottengo lo stato che sarebbero le osservazioni (immagine 84x96x3)
        state = self.get_observation()

        reward = 1 # inutile
        
        # quando il numero di step super quello degli episodi, done diventa true e si conclude l'episodio
        if self.current_steps > self.total_episode_steps:
            done = True

        return state, reward, done



    def close(self):
        """
        Funzione che permette di chiudere la comunicazione con PyBullet
        """
        p.disconnect()



    def getCamera_image(self):
        """
        Funzione che estrapola l'immagine BGR dalla camera posta sul veicolo.
        
        Returns:
        - imgHSV: immagine nel formato colore HSV
        """
        camInfo = p.getDebugVisualizerCamera()
        ls = p.getLinkState(self.car_id,zed_camera_joint, computeForwardKinematics=True)
        camPos = ls[0]
        camOrn = ls[1]
        camMat = p.getMatrixFromQuaternion(camOrn)
        forwardVec = [camMat[0],camMat[3],camMat[6]]
        camUpVec =  [camMat[2],camMat[5],camMat[8]]
        camTarget = [camPos[0]+forwardVec[0]*10,camPos[1]+forwardVec[1]*10,camPos[2]+forwardVec[2]*10]
        viewMat = p.computeViewMatrix(camPos, camTarget, camUpVec)
        projMat = camInfo[3]
        
        width, height, rgbImg, depthImg, segImg= p.getCameraImage(640,480,viewMatrix=viewMat,projectionMatrix=projMat, renderer=p.ER_BULLET_HARDWARE_OPENGL)

        imgHSV = cv2.cvtColor(rgbImg[:,:,:3], cv2.COLOR_RGB2HSV)
        return imgHSV


    
    def getVelocity(self):
        """
        Funzione che calcola la velocità rispetto al frame locale della macchina.
        
        Returns:
        - linear_velocity: velocità lineare in terna locale
        - angular_velocity: velocità angolare in terna locale
        
        """
        linearVelocity, angularVelocity = p.getBaseVelocity(self.car_id)
        base_pose =  p.getBasePositionAndOrientation(self.car_id)

        # Converto le velocità angolari e lineari in local frame
        rotation_matrix = p.getMatrixFromQuaternion(base_pose[1])  # Rotation matrix from quaternion
        rotation_matrix = np.reshape(rotation_matrix, (3, 3))

        # Velocità lineare nel frame locale
        local_linear_velocity = np.dot(np.linalg.inv(rotation_matrix), np.array(linearVelocity))

        # Velocità angolare nel frame locale
        local_angular_velocity = np.dot(np.linalg.inv(rotation_matrix), np.array(angularVelocity))
        
        # Estrazione valori di velocità
        linear_velocity = np.array(local_linear_velocity)[0] # x velocity
        angular_velocity = np.array(local_angular_velocity)[2] # z angular

        return linear_velocity, angular_velocity
    


    def stoppingCar(self):
        """
        Funzione che ferma il veicolo una volta terminata la simulazione.
        """
        steer = 0
        forward = 0
        
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
        """
        Funzione che fornisce posizione e orientazione della macchina.
        
        Returns:
        - car_position: posizione della macchina in terna global
        - car_orientation: orientazione della macchina in terna global
        
        """
        car_position, car_orientation = p.getBasePositionAndOrientation(self.car_id)
        
        return car_position, car_orientation
    
    
    
    def wrap2pi(self,angle):
        """
        Funzione che mappa l'angolo tra +- pi.
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi



    def rect_to_polar_relative(self,goal):
        """
        Funzione usata per la trasformazione in coordinate polari del goal
        
        Parameters:
        - goal: punto per il quale bisogna calcolare le coordinate polari
        
        Returns:
        - r: raggio, ovvero la distanza tra veicolo e goal
        - theta: angolo di orientazione del veicolo rispetto al goal
        """
        
        # calcolo la posizione corrente del veicolo
        car_position, car_orientation = self.getCarPosition()
        
        # calcolo l'angolo di yaw
        _,_,yaw = p.getEulerFromQuaternion(car_orientation)
        
        # calcolo delle coordinate polari
        vector_to_goal = np.array([goal[0] - car_position[0], goal[1] - car_position[1]])
        r = np.linalg.norm(vector_to_goal)
        theta = self.wrap2pi(np.arctan2(vector_to_goal[1], vector_to_goal[0])-self.wrap2pi(yaw))
        return r, theta



    def choosePositionAndIndex(self,position,index):
        """
        Funzione che calcola la posizione da cui partire (in termini di coordinate) e il corrispondente indice della lista dei punti intermedi.
        
        Parameters:
        - position: coordinate del punto prescelto per la partenza
        - index: indice della lista dei punti intermedi di quel punto
        
        Returns:
        - positionToStart: posizione effettivamente scelta per la partenza
        - indexToStart: indice relativo al punto effettivamente scelto
        """
        if len(position!=0):
            for i,j in zip(range(len(position)),position):
                r, yaw_error = self.rect_to_polar_relative(j[:2])
                vel_ang,_ = self.p_control(yaw_error)
                
                # se trovo un punto di partenza con una velocità angolare troppo elevata
                # significa che il punto è dietro di me e quindi lo scarto
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
    
    

    def p_control(self,yaw_error):
        """
        Funzione che definisce il controllore P.
        
        Parameters:
        - yaw_error: angolo di orientazione del veicolo rispetto al goal
        
        Returns:
        - vel_ang: velocità angolare da comandare al veicolo
        - vel_lin: velocità lineare da comandare al veicolo
        """
        kp = 0.9
        vel_ang = kp*yaw_error
        vel_lin = 10 # m/s
    
        if abs(yaw_error)>0.1:
            vel_lin = (2.3-abs(vel_ang))*(10/2.3)
        return vel_ang, vel_lin
    


    def getHomographyFromPts(self, greenMask):
        """
        Funzione che calcola l'omografia utile a passare in vista BirdEye.
        
        Returns:
        - HomoMat: matrice 3x3 perla trasformazione omografica
        """
        if self.HomoMat is not None:
            return self.HomoMat
        
        # trovo in automatico i punti sorgenti per la bird eye: dove i pixel sono neri -> 255
        findEdge = lambda x, line: np.argwhere(np.abs(np.diff(x[line,:].astype(np.int16)))==255)
        topEdge = findEdge(greenMask,320).flatten()
        botEdge = findEdge(greenMask, 460).flatten()
        
        # calcolo l'omografia una sola volta
        # if self.HomoMat is None and len(topEdge)==2 and len(botEdge)==2:
        #     origPts = np.array([[topEdge[0], 320],[topEdge[1], 320],[botEdge[1], 460],[botEdge[0], 460]]).astype(np.float32)
        #     destPts = np.array([[250,360],[390,360],[390,460],[250,460]]).astype(np.float32)
        #     self.HomoMat =cv2.getPerspectiveTransform(origPts, destPts)
        
        self.HomoMat = np.array([[-2.98962782e-01, -1.24344112e+00,  3.93075046e+02],
            [-3.38241831e-15, -2.16351434e+00,  5.70860281e+02],
            [-8.01126080e-18, -4.17937767e-03,  1.00000000e+00]])
        return self.HomoMat


    
    def birdEyeView(self,imgHSV):
        """
        Funzione che permette di mappare i punti della strada dalla vista frontale a quella birdEye rappresentando
        la corsia di bianco e tutto il resto di nero.
        
        Parameters:
        - imgHSV: immagine in formato colore HSV
        
        Returns:
        - reshapedBird: birdEye 84x96x1 
        """
        
        skyMsk = cv2.inRange(imgHSV[:,:,0], 115,125) # maschera del cielo
        greenMsk = cv2.inRange(imgHSV[:,:,0], 55,65) # maschera del prato
    
        # rimuovo il rumore con un filtro di dimensione 3x1
        greenMsk = cv2.morphologyEx(greenMsk, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(1,3)), iterations=2)
        skyMsk = cv2.morphologyEx(skyMsk, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(1,3)), iterations=4)
        
        # calcolo dell'omografia
        self.HomoMat = self.getHomographyFromPts(greenMsk)
    
        # birdeye togliendo il cielo e il prato
        # risultato: cielo e prato neri e la strada bianca
        # la funzione cv2.warpPerspective viene utilizzata per applicare la Perspective Transform ad un'immagine,
        # data la matrice di trasformazione
        # INPUT:
        # input image
        # transformation matrix
        # dimensioni immagine finale
        # metodo di interpolazione
        bird_eye = cv2.warpPerspective(255-(greenMsk+skyMsk), self.HomoMat, (640,480),flags=cv2.INTER_LINEAR)[::3,::3]
        
        reshapedBird = cv2.resize(bird_eye, (96, 84)) # larghezza, altezza
   
        return reshapedBird



    def computeIndexToStart(self, centerLine):
        """
        Funzione che restituisce la lista completa e riordinata della linea centrale: l'inzio è la posizione di partenza 
        della macchina
        
        Parameters:
        - centerLine: punti della linea intermedia tra i coni
        
        Returns:
        - total_index: lista riordinata di indici 
        """
        positionToStart = []
        threshold = 0.25
        car_position, _ = self.getCarPosition()
    
        # lista delle possibili posizioni di partenza: se non trovo un punto di partenza incremento la threshold
        while len(positionToStart)==0:
            index,position = getPointToStart(centerLine,car_position[:2], threshold)
            positionToStart, indexToStart = self.choosePositionAndIndex(position,index)
            threshold+=0.05
                
        
        # creo le liste di indici 
        indexList_1 = [idx for idx in range(indexToStart,len(centerLine),1)]
        indexList_2 = [idx for idx in range(0,indexToStart,1)]
        
        total_index = indexList_1+indexList_2
        
        return total_index
import random
import time
import gym
import numpy as np
import pybullet as p
import pybullet_data
import cv2
from get_track import *
from ultralytics import YOLO


zed_camera_joint = 7 # simplecar



class PyBulletContinuousEnv(gym.Env):
    def __init__(self, total_episode_step=1999):
        super(PyBulletContinuousEnv, self).__init__()

        # Connessione a PyBullet e setup della simulazione
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        #p.setRealTimeSimulation(1)

        # numero totale di episodi
        self.total_episode_steps = total_episode_step
        # matrice per l'omografia
        self.HomoMat = None
        # inizializzo il modello per la predizione dei coni
        self.model = YOLO('best.pt','gpu')


    # funzione utile a inizializzare l'ambiente 
    def reset(self):
        self.current_steps = 0
        
        # Reset della simulazione
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        train = False
                
        track_list_train = [0,1,2,4,5] # lista degli ID dei tracciati
        #track_list_train = [5] #prova con modello singolo
        track_list_test = [6]
        
        if train:
            self.track_number = random.choice(track_list_train)
        else:
            self.track_number = random.choice(track_list_test)
        
        
        #p.resetDebugVisualizerCamera(cameraDistance=20, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[5,10,0])
        
        p.resetDebugVisualizerCamera(cameraDistance=25, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[0,-2,0])
 
        # carico il tracciato
        p.loadURDF("world&car/plane/plane.urdf")
   
        # punti di spawn della macchina per la road
        # env1 = [[7,-4,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(180)])]
        # env2 = [[7,20,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(0)])]
        # env3 = [[-7.6,6,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(90)])]
        # env4 = [[7.3,13.9,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(145)])]
        
        
        # env_list = [env1,env2,env3,env4]
        # index_env = random.randint(0,len(env_list)-1)
        #self.car_id = p.loadURDF("world&car/simplecar.urdf", env_list[index_env][0],env_list[index_env][1])

        
        # TRACK 0
        if self.track_number==0:
            self.car_id = p.loadURDF("world&car/simplecar.urdf", [-9.93,8.84,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(90)]))
        
        elif self.track_number==1:
        # TRACK 1
            self.car_id = p.loadURDF("world&car/simplecar.urdf",[12.22, -6.26,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(-125)]))
            
        elif self.track_number==2:
        # TRACK 2
            self.car_id = p.loadURDF("world&car/simplecar.urdf",[7.32, -16.6,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(180)]))
        
        elif self.track_number==4:
        # TRACK 4
            self.car_id = p.loadURDF("world&car/simplecar.urdf",[-11.9, 10.23,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(45)]))
        
        elif self.track_number==5:
        # TRACK 5
            self.car_id = p.loadURDF("world&car/simplecar.urdf",[1.27, -6.78,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(115)]))
        
        elif self.track_number==6:
        # TRACK 6
            #self.car_id = p.loadURDF("world&car/simplecar.urdf",[-17.2, 10.91,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(45)]))
            self.car_id = p.loadURDF("world&car/simplecar.urdf",[-19.8, 12.8,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(45)]))
        
        elif self.track_number==7:
        # TRACK 7
            #self.car_id = p.loadURDF("world&car/simplecar.urdf",[-13.5, 9.43,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(30)]))
            self.car_id = p.loadURDF("world&car/simplecar.urdf",[-20.1, 7,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(30)]))
        
        elif self.track_number==9:
        # TRACK 9
            self.car_id = p.loadURDF("world&car/simplecar.urdf",[6.7, 18.8,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(0)]))
        
        
        
        p.loadSDF(f"world/track/track{self.track_number}.sdf",globalScaling = 1)


    # Le osservazioni sono le immagini 84x96x1 rgb prese dalla zed
    def get_observation(self):
        
        rgb_image = self.getCamera_image()
        
        # eseguo la predizione ed ottengo info sui bounding box e sulle classi trovate
        boxes_xywh, boxes_cls = self.conesPrediction(rgb_image)
        
        # funzione che, date le info sui box, mi restituisce tutti i centri dei coni blu e gialli
        blue_center, yellow_center = self.divide_centers(boxes_xywh,boxes_cls)
        
        # funzione che mi definisce la bird eye dopo aver eseguito una serie di controllo per eliminare i coni lontani 
        reshaped_bird_eye = self.birdEyeView_cones(blue_center, yellow_center)
        
        
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

        #p.setTimeStep(0.02)
        # per andare a 10Hz
        for _ in range(24):
            p.stepSimulation()

        # Ottengo lo stato che sarebbero le ossservazioni (immagine 84x96x1)
        state = self.get_observation()

        reward = 1 # per ora lascio 1
        
        # quando il numero di step super quello degli episodi, done diventa true e si conclude l'episodio
        if self.current_steps > self.total_episode_steps:
            done = True

        return state, reward, done


    # chiudo tutto
    def close(self):
        p.disconnect()


    # ottengo l'immagine dalla zed montata sul robot e ritorno l'immagine RGB 640X480x3
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

        return rgbImg[:,:,:3]


    # funzione che esegue la predizione del modello di classificazione
    def conesPrediction(self, img):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        results = self.model.predict(img,conf=0.8)
        boxes_xywh = results[0].boxes.xywh  # boxes mi da xywh di tutti i coni visti
        boxes_cls = results[0].boxes.cls  # boxes mi da le classi di tutti i coni visti
        
        #annotated_frame = results[0].plot()
        
        return boxes_xywh, boxes_cls


    def divide_centers(self,boxes_xywh, boxes_cls):
        yellow_center,blue_center = [],[]

        for box,cls in zip(boxes_xywh,boxes_cls):
            if cls==0:
                x_blue, y_blue, w, h = box
                blue_center.append([x_blue.cpu().item(),y_blue.cpu().item()])
            elif cls==4:
                x_yellow, y_yellow, w, h = box
                yellow_center.append([x_yellow.cpu().item(),y_yellow.cpu().item()])

        return np.array(blue_center), np.array(yellow_center)



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
    

    def getHomographyFromPts(self):
        if self.HomoMat is not None:
            return self.HomoMat
        # scelta dei 4 punti nell'immagine di partenza da mappare poi nei punti di destinazione dst dell'immagine finale
        # questi 4 punti sono la regione d'interesse
        # basso sinistra, alto sinistra, alto destra, basso destra
        
        #conePts4Homography = np.array([[186,324], [458,324], [601,412], [43,412]]).astype(np.float32)
        
        #conePts4Homography = np.array([[155,317], [489,317], [579,371], [86,371]]).astype(np.float32)
        #                                  UL        UR          LR        LL
        #coneDesiredPts     = np.array([[ 140,307], [500,307], [500,400], [140,400]  ]).astype(np.float32)
        # 155,317     489,317         579,371       86,371 
       
        #HomoMat = cv2.getPerspectiveTransform(conePts4Homography,coneDesiredPts)
        #print(HomoMat)
       
        self.HomoMat = np.array([[   -0.46095,     -1.3316,      468.43],
                             [ 9.4581e-16,     -2.0326,      551.64],
                             [ 2.3918e-18,      -0.0041613,       1]])
        return self.HomoMat



    def sortedPoints(self, blue_center:np.array, yellow_center:np.array):
       
        # riordino l'array
        # la colonna prima è quella con priorità più bassa -> riordino lungo le y
        sorted_indices_b = np.lexsort((blue_center[:, 0], blue_center[:, 1]))
        sorted_indices_y = np.lexsort((yellow_center[:, 0], yellow_center[:, 1]))
       
        pts_blue = blue_center[sorted_indices_b].astype(int)
        pts_yellow = yellow_center[sorted_indices_y].astype(int)
        
        return pts_blue, pts_yellow


    def birdEyeView_cones(self,blue_center: np.array, yellow_center: np.array):
        
        # funzione che esegue l'interpolazione
        def getLine(cones):
            """Torna un insieme di punti campione a partire da punti sparsi nell'immagine"""
            len=cones.shape[0]
            ii=scipy.interpolate.interp1d(np.arange(len), cones.squeeze(), axis=0, kind='linear',bounds_error=False, fill_value='extrapolate')
            pts=ii(np.arange(-len,2*len,.25))
            return pts.reshape(-1,1,2).astype(np.int32)
        
        # ottengo la matrice utile ad eseguire l'omografia
        HomoMat = self.getHomographyFromPts()


        # aggiungo un cono blu o uno giallo ai lati dell'immagine quando siamo in curva  
        if blue_center.shape[0] == 0:
            blue_center = np.array([[0,479]], dtype=int)

        if yellow_center.shape[0] == 0:
            yellow_center = np.array([[639,479]], dtype=int)
       
        # eseguo il sorted lungo la y dei coni
        pts_blue, pts_yellow = self.sortedPoints(blue_center, yellow_center)

        # Trasforma la posizione dei coni in vista verticale  
        # per ogni punto applico la matrice di trasformazione per ottenere il punto mappato nella vista verticale     
        applyHomo = lambda pts: cv2.convertPointsFromHomogeneous((HomoMat@cv2.convertPointsToHomogeneous(pts)[:,0,:].T).T).astype(int)
        pts_bird_b = applyHomo(pts_blue)
        pts_bird_y = applyHomo(pts_yellow)
    
       
        # Dopo la vista verticale posso eliminare altri coni...
        # elimino i coni che sono molto distanti dal centro immagine ovvero dal centro camera
        dist = lambda p : abs(p[0,0]-320)+abs(p[0,1]-480)
        select = lambda v: v[np.array([dist(ve) for ve in v])<980]
        pts_bird_b = select(pts_bird_b)
        pts_bird_y = select(pts_bird_y)
       
        # inizializzo l'immagine che mostrerà la birdeye
        bird = np.zeros((480,640), dtype=np.uint8)


        # Eseguo una serie di controlli per calcolare l'indice del primo elemento appena dentro la dimensione dell'immagine (sia per il blue
        # che per il giallo): questo mi permette di andare a definire un "point" che sarebbe quello da passare al floodFill in quanto questa 
        # funzione richiede un punto all'interno dell'immagine per poter riempire il poligono
        # la dimensione è 0 quando yellow center non è vuoto ma il punto è troppo lontano e quindi con il select viene tolto
        if pts_bird_b.shape[0]>1:
            cv2.polylines(bird, [getLine(pts_bird_b)], isClosed=False,color=(255), thickness=1, lineType=cv2.LINE_AA)
            
            id_b = np.where((getLine(pts_bird_b)[:,0,0]>=0) & (getLine(pts_bird_b)[:,0,0]<=640) & (getLine(pts_bird_b)[:,0,1]>=0) & (getLine(pts_bird_b)[:,0,1]<=480))[0]
            
            # può essere che ci siano punti negativi e anche con una y elevata e quindi setto il nuovo punto
            if id_b.shape[0]==0:
                point = (0, 479)
            else:
                id_b = id_b
        else:
            id_b = np.array([])
            point = (0, 479)
   
        if pts_bird_y.shape[0]>1:
            cv2.polylines(bird, [getLine(pts_bird_y)], isClosed=False,color=(255), thickness=1, lineType=cv2.LINE_AA)
            
            id_y = np.where((getLine(pts_bird_y)[:,0,0]>=0) & (getLine(pts_bird_y)[:,0,0]<=640) & (getLine(pts_bird_y)[:,0,1]>=0) & (getLine(pts_bird_y)[:,0,1]<=480))[0]
       
            if id_y.shape[0]==0:
                point = (639, 479)
        else:
            id_y = np.array([])
            point = (639, 479)
                
                  
        if id_b.shape[0]!=0 and id_y.shape[0]!=0:
            point = (int((getLine(pts_bird_y)[id_y[-1]].squeeze()[0]+getLine(pts_bird_b)[id_b[-1]].squeeze()[0])/2),479)
      
        
        # funzione che mi riempie un poligono dato un punto che sta tra due linee
        cv2.floodFill(bird,None,point,255)
        
        # resize immagine
        reshapedBird = cv2.resize(bird, (96, 84)) # larghezza, altezza
        
        return reshapedBird



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
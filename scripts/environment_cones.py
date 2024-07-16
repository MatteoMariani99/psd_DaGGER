import random
import gym
import numpy as np
import pybullet as p
import pybullet_data
import cv2
from get_track import *
from ultralytics import YOLO

# giunto relativo alla camera sul veicolo "simplecar"
zed_camera_joint = 7 # simplecar



class ConesEnv(gym.Env):
    def __init__(self, total_episode_step=1999):
        super(ConesEnv, self).__init__()

        # Connessione a PyBullet e setup della simulazione
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.total_episode_steps = total_episode_step # numero totale di episodi
        self.HomoMat = None # matrice per l'omografia
        self.model = YOLO('best.pt','gpu') # inizializzo il modello per la predizione dei coni


    
    def reset(self):
        """
        Funzione che permette di inizializzare l'ambiente (tracciato e veicolo)
        """
        self.current_steps = 0
        
        # Reset della simulazione
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        train = False # se train uguale false, allora consideriamo i tracciato per il testing
                
        track_list_train = [0,1,2,4,5] # lista degli ID dei tracciati
        track_list_test = [6]
        
        if train:
            self.track_number = random.choice(track_list_train)
        else:
            self.track_number = random.choice(track_list_test)
        
        
        # posizionamento camera di Pybullet per la vista completa del tracciato
        p.resetDebugVisualizerCamera(cameraDistance=25, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[0,-2,0])
 
        # caricamento del piano
        p.loadURDF("world/models/plane/plane.urdf")
   
        # TRACK 0
        if self.track_number==0:
            self.car_id = p.loadURDF("world/models/car/simplecar.urdf", [-9.93,8.84,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(90)]))
        
        # TRACK 1
        elif self.track_number==1:
            self.car_id = p.loadURDF("world/models/car/simplecar.urdf",[12.22, -6.26,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(-125)]))
        
        # TRACK 2
        elif self.track_number==2:
            self.car_id = p.loadURDF("world/models/car/simplecar.urdf",[7.32, -16.6,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(180)]))
        
        # TRACK 4
        elif self.track_number==4:
            self.car_id = p.loadURDF("world/models/car/simplecar.urdf",[-11.9, 10.23,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(45)]))
        
        # TRACK 5
        elif self.track_number==5:
            self.car_id = p.loadURDF("world/models/car/simplecar.urdf",[1.27, -6.78,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(115)]))
        
        # TRACK 6
        elif self.track_number==6:
            self.car_id = p.loadURDF("world/models/car/simplecar.urdf",[-19.8, 12.8,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(45)]))
        
        # TRACK 7
        elif self.track_number==7:
            self.car_id = p.loadURDF("world/models/car/simplecar.urdf",[-20.1, 7,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(30)]))
        
        # TRACK 9
        elif self.track_number==9:
            self.car_id = p.loadURDF("world/models/car/simplecar.urdf",[6.7, 18.8,.3], p.getQuaternionFromEuler([0,0,np.deg2rad(0)]))
        
        
        # caricamento del tracciato selezionato
        p.loadSDF(f"world/models/track/track{self.track_number}.sdf",globalScaling = 1)



    def get_observation(self):
        """
        Funzione che restituisce le osservazioni, ovvero immagini 84x96x1 birdEye, utili all'algoritmo.
        
        Returns:
            - reshaped_bird_eye: immagini birdEye con resize a 84x96x1
        """
        bgr_image = self.getCamera_image()
        
        # eseguo la predizione ed ottengo info sui bounding box e sulle classi trovate
        boxes_xywh, boxes_cls = self.conesPrediction(bgr_image)
        
        # funzione che, date le info sui box, mi restituisce tutti i centri dei coni blu e gialli
        blue_center, yellow_center = self.divide_centers(boxes_xywh,boxes_cls)
        
        # funzione che mi definisce la bird eye dopo aver eseguito una serie di controlli per eliminare i coni lontani 
        reshaped_bird_eye = self.birdEyeView_cones(blue_center, yellow_center)
        
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

        state = self.get_observation() # ottengo le immagini birdEye 84x96x1

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
        - bgrImg: immagine frontale BGR 640x480x3
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
        
        width, height, bgrImg, depthImg, segImg= p.getCameraImage(640,480,viewMatrix=viewMat,projectionMatrix=projMat, renderer=p.ER_BULLET_HARDWARE_OPENGL)

        return bgrImg[:,:,:3]


    
    def conesPrediction(self, img):
        """
        Funzione che esegue la predizione dei coni tramite il modello YoloV8.
        
        Parameters:
        - img: immagine BGR su cui identificare i coni.
        
        Returns:
        - boxes_xywh: centro in coordinate xy del cono e altezza/larghezza dal centro.
        - boxes_cls: classe del cono relativa a quel bounding box identificato
        
        """
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # trasformo l'immagine in RGB
        
        results = self.model.predict(img,conf=0.8) # identifico solo i coni che hanno una probabilità di esserlo maggiore dell'80%
        boxes_xywh = results[0].boxes.xywh  # boxes mi da xywh di tutti i coni visti
        boxes_cls = results[0].boxes.cls  # boxes mi da le classi di tutti i coni visti
        
        #annotated_frame = results[0].plot() # per fare il plot dei coni + bounding box
        
        return boxes_xywh, boxes_cls



    def divide_centers(self,boxes_xywh, boxes_cls):
        """
        Funzione permette di calcolare il centro per ciascun cono giallo e blu.
        
        Parameters:
        - boxes_xywh: centro in coordinate xy del cono e altezza/larghezza dal centro.
        - boxes_cls: classe del cono relativa a quel bounding box identificato
        
        Returns:
        - blue_center: elenco dei centri dei coni blu
        - yellow_center: elenco dei centri dei coni gialli
        
        """
        yellow_center,blue_center = [],[]

        for box,cls in zip(boxes_xywh,boxes_cls):
            if cls==0:
                x_blue, y_blue, w, h = box
                blue_center.append([x_blue.cpu().item(),y_blue.cpu().item()])
            elif cls==4:
                x_yellow, y_yellow, w, h = box
                yellow_center.append([x_yellow.cpu().item(),y_yellow.cpu().item()])

        return np.array(blue_center), np.array(yellow_center)



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
    


    def getHomographyFromPts(self):
        """
        Funzione che calcola l'omografia utile a passare in vista BirdEye.
        
        Returns:
        - HomoMat: matrice 3x3 perla trasformazione omografica
        """
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
       
        self.HomoMat = np.array([[   -0.46095,     -1.3316,      468.43],
                             [ 9.4581e-16,     -2.0326,      551.64],
                             [ 2.3918e-18,      -0.0041613,       1]])
        return self.HomoMat



    def sortedPoints(self, blue_center:np.array, yellow_center:np.array):
        """
        Funzione che riordina i centri dei coni lungo la coordinata y.
        
        Parameters:
        - blue_center: centro dei coni blu
        - yellow_center: centro dei coni gialli
        
        Returns:
        - pts_blue: centri dei coni blu riordinati 
        - pts_yellow: centri dei coni gialli riordnati
        """
        # riordino l'array
        # la colonna prima è quella con priorità più bassa -> riordino lungo le y
        sorted_indices_b = np.lexsort((blue_center[:, 0], blue_center[:, 1]))
        sorted_indices_y = np.lexsort((yellow_center[:, 0], yellow_center[:, 1]))
       
        pts_blue = blue_center[sorted_indices_b].astype(int) # cast a int in quanto sono pixel
        pts_yellow = yellow_center[sorted_indices_y].astype(int) # cast a int in quanto sono pixel
        
        return pts_blue, pts_yellow



    def birdEyeView_cones(self,blue_center: np.array, yellow_center: np.array):
        """
        Funzione che permette di mappare i centri dei coni dalla vista frontale a quella birdEye: inoltre 
        costruisce il poligono che rappresenta la corsia.
        
        Parameters:
        - blue_center: centro dei coni blu
        - yellow_center: centro dei coni gialli
        
        Returns:
        - reshapedBird: birdEye 84x96x1 
        """
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
    
       
        # Dopo la vista verticale posso eliminare altri coni.
        # elimino i coni che sono molto distanti dal centro immagine ovvero dal centro camera
        dist = lambda p : abs(p[0,0]-320)+abs(p[0,1]-480)
        select = lambda v: v[np.array([dist(ve) for ve in v])<980]
        pts_bird_b = select(pts_bird_b)
        pts_bird_y = select(pts_bird_y)
       
        # inizializzo l'immagine che mostrerà la birdeye
        bird = np.zeros((480,640), dtype=np.uint8)


        # Eseguo una serie di controlli per calcolare l'indice del primo elemento appena dentro la dimensione dell'immagine (sia per il blu
        # che per il giallo): questo mi permette di andare a definire un "point" che sarebbe quello da passare al floodFill in quanto questa 
        # funzione richiede un punto all'interno dell'immagine per poter riempire il poligono.
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
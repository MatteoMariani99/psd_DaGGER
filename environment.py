import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import cv2
from itertools import chain
import torch


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
        p.loadSDF("f10_racecar/meshes/barca_track.sdf", globalScaling=1)
        
   

        self.car_id = p.loadURDF("f10_racecar/simplecar.urdf", [-9,-6.5,.3])
        #print()
        #print(p.getEulerFromQuaternion(p.getQuaternionFromEuler([0,0,100]))[2]*180/(2*3.14))
        # sphere_color = [1, 0, 0,1]  # Red color
        # sphere_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=0)
        # sphere_body = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=sphere_shape,basePosition=[-10,8, 0])
        # p.changeVisualShape(sphere_body, -1, rgbaColor=sphere_color)


    # Le osservazioni sono le immagini 96x96 rgb prese dalla zed
    def get_observation(self):
        
        velocity_vector = []
        pose_vector = []

        # Nel caso in cui servano posizione e velocità rispetto al frame world
        car_position, car_orientation = p.getBasePositionAndOrientation(self.car_id)
        car_velocity_x, car_angular_velocity_z = self.getVelocity()

        _,_,yaw = p.getEulerFromQuaternion(car_orientation)
        #print(yaw*180/(2*3.14))

        velocity_vector.append([car_velocity_x, car_angular_velocity_z])
        pose_vector.append([car_position[0], car_position[1],yaw])

        #print("Velocità(x) e angulare(z): ",velocity_vector)
        #print("Posizione e yaw: ",pose_vector)
        rgb_image = self.getCamera_image()

        # immagine in formato YUV
        return rgb_image

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
        #for _ in range(24):
        p.stepSimulation()

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
        
        # ottengo le 3 immagini: rgb, depth, segmentation
        width, height, rgbImg, depthImg, segImg= p.getCameraImage(200,66,viewMatrix=viewMat,projectionMatrix=projMat, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        # faccio un reshape in quanto da sopra ottengo un array di elementi
        rgb_opengl = np.reshape(rgbImg, (height, width, 4)) 

        # Tolgo il canale alpha e converto da BGRA a RGB per la rete
        rgb_image = cv2.cvtColor(rgb_opengl, cv2.COLOR_BGRA2RGB)
        #print("Reshaped RGB image size:", rgb_image.shape)

        return rgb_image

        # Per vedere le immagini in opencv
        # cv2.imshow("Camera", rgb_image)
        # cv2.waitKey(0) 
    

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




    # # funzione che permette di ottenere la polica a dell'expert
    # # per inizializzare li metto nella chiamata
    # def getAction_expert(self,turn=0,forward=0,backward=0):
    #     p.setGravity(0,0,-10)
    #     keys = p.getKeyboardEvents()

    #     # comandi da tastiera per l'expert
    #     for k,v in keys.items():

    #             if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
    #                     turn = 0.5
    #             if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_RELEASED)):
    #                     turn = 0
    #             if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
    #                     turn = -0.5
    #             if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_RELEASED)):
    #                     turn = 0

    #             if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_TRIGGERED)):
    #                     forward=20
    #             if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_RELEASED)):
    #                     forward=0
    #             if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_TRIGGERED)):
    #                     backward=20
    #             if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_RELEASED)):
    #                     backward=0

    #     # la riga 161 del dagger.py deve essere sostituita con questa
    #     action = np.array([turn, forward, backward]).astype('float32')

    #     return action
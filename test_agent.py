import numpy as np
import draw_steering_angle
import torch
import cv2
from model import Model
import math
import time

from environment_cones import PyBulletContinuousEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Torch Device:", device)

# punti per volante e barra verticale
pts = np.array([[0, 0], [480, 0],
        [480, 70], [0, 70]],
        np.int32)

pts = pts.reshape((-1, 1, 2))

def run_episode(max_timesteps=1000):
    
    #episode_reward = 0
    step = 0
    env.reset()
    state= env.get_observation()
    
    while True:
        start = time.time()
        
        color_rgb = env.getCamera_image()
        bird_eye = cv2.resize(color_rgb, (480, 320))

        cv2.fillPoly(bird_eye, pts=[pts], color=(0, 0, 0))

        steering_wheel = draw_steering_angle.SteeringWheel(bird_eye)
    
        prediction = agent(torch.from_numpy(state[np.newaxis,np.newaxis,...]).type(torch.FloatTensor).to(device))
        # np.newaxis aumenta la dimensione dell'array di 1 (es. se è un array 1D diventa 2D)
        # torch.from_numpy crea un tensore a partire da un'array numpy
        # il modello ritorna le azioni (left/right, up/down)

        a = prediction.detach().cpu().numpy().flatten()
           
        # # disegno il volante per lo sterzo 
        steering_wheel.draw_steering_wheel_on_image(a[0]*180/math.pi,(20,10))
        # aggiungo la barra verticale per la velocità
        vel_image = steering_wheel.update_frame_with_bar(a[1])
        
        text = " rad/s"
        full_text = f"{str(round(a[0],3))}{text}" 
        text1 = " m/s"
        full_text1 = f"{str(round(a[1],2))}{text1}" 
        
        # Display del testo a video
        cv2.putText(vel_image, full_text, (90,40), cv2.FONT_HERSHEY_SIMPLEX,  
                        0.6, (255,255,255), 1, cv2.LINE_AA) 
        cv2.putText(vel_image, full_text1, (360,40), cv2.FONT_HERSHEY_SIMPLEX,  
                        0.6, (255,255,255), 1, cv2.LINE_AA) 

        #image_obs = state
        image_obs = cv2.resize(state, (480, 320))
        image_obs = cv2.cvtColor(image_obs, cv2.COLOR_GRAY2RGB)

        cv2.imshow("Camera2", cv2.vconcat([vel_image, image_obs]))
        cv2.waitKey(1) 

        
        # take action, receive new state & reward
        #a = [0,0]
        
        next_state, _, done = env.step(a)
        print("-----seconds-----", time.time()-start)
        #cv2.imshow("Camera",next_state)
        
        
        #episode_reward += reward       
        state = next_state
        step += 1

        if done or step > max_timesteps: 
            break
        print("-----seconds-----", time.time()-start)

    #return episode_reward


if __name__ == "__main__":                
    
    # numero di episodi 
    n_test_episodes = 15                  

    # se voglio fargli fare il giro della pista basta che modifico le iterazioni nell env
    env = PyBulletContinuousEnv()
    #env.reset()
    # istanza del modello
    agent = Model()
    # state= env.get_observation()
    # print(state.shape)
    # # Export the model
    # x = torch.randn(1, 1, 84,96)
    # torch.onnx.export(agent,               # model being run
    #                 x,                         # model input (or a tuple for multiple inputs)
    #                 "model.onnx",   # where to save the model (can be a file or file-like object)
    #                 export_params=True)

    # carico il modello ottimo ottenuto
    #agent.load("dagger_test_models/modelli ottimi/vel10_variabile.pth")
    agent.load("dagger_test_models/modelli ottimi/cones/vel10_variabile.pth")
    agent.to(device)

  
    for i in range(n_test_episodes):
        run_episode()
        
   
            
    env.close()
    print('... finished')



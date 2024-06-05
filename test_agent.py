import numpy as np
import draw_steering_angle
import torch
import cv2
from model import Model
import math


from environment import PyBulletContinuousEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Torch Device:", device)



def run_episode(env:PyBulletContinuousEnv, agent, max_timesteps=1000000):
    
    #episode_reward = 0
    step = 0
    env.reset()

    while True:

        state= env.get_observation()
        colorHSV = env.getCamera_image()
        bird_eye = cv2.resize(colorHSV, (480, 320))
        
        # punti per volante e barra verticale
        pts = np.array([[0, 0], [480, 0],
                [480, 70], [0, 70]],
               np.int32)
 
        pts = pts.reshape((-1, 1, 2))

        cv2.fillPoly(bird_eye, pts=[pts], color=(0, 0, 0))

        steering_wheel = draw_steering_angle.SteeringWheel(bird_eye)
    
        prediction = agent(torch.from_numpy(state[np.newaxis,np.newaxis,...]).type(torch.FloatTensor).to(device))
        # np.newaxis aumenta la dimensione dell'array di 1 (es. se è un array 1D diventa 2D)
        # torch.from_numpy crea un tensore a partire da un'array numpy
        # il modello ritorna le azioni (left/right, up/down)

        a = prediction.detach().cpu().numpy().flatten()
           
        # disegno il volante per lo sterzo 
        steering_wheel.draw_steering_wheel_on_image(a[0]*180/math.pi,(20,10))
        # aggiungo la barra verticale per la veocità
        vel_image = steering_wheel.update_frame_with_bar(a[1])
        
        text = " rad/s"
        full_text = f"{str(round(a[0],3))}{text}" 
        text1 = " m/s"
        full_text1 = f"{str(round(a[1],2))}{text1}" 
        
        # Display del testo a video
        final_image = cv2.putText(vel_image, full_text, (90,40), cv2.FONT_HERSHEY_SIMPLEX,  
                        0.6, (255,255,255), 1, cv2.LINE_AA) 
        final_image = cv2.putText(final_image, full_text1, (360,40), cv2.FONT_HERSHEY_SIMPLEX,  
                        0.6, (255,255,255), 1, cv2.LINE_AA) 

        image_obs = state
        image_obs = cv2.resize(image_obs, (480, 320))
        image_obs = cv2.cvtColor(image_obs, cv2.COLOR_GRAY2BGR)

        cv2.imshow("Camera2", cv2.vconcat([final_image, image_obs]))
        cv2.waitKey(1) 

        
        # take action, receive new state & reward
        next_state, reward, done = env.step(a)
        
        #episode_reward += reward       
        state = next_state
        step += 1

        if done or step > max_timesteps: 
            break

    #return episode_reward


if __name__ == "__main__":                
    
    # numero di episodi 
    n_test_episodes = 15                  

    # istanza del modello
    agent = Model()

    # carico il modello ottimo ottenuto
    agent.load("dagger_test_models/modelli ottimi/vel10_variabile.pth")
    agent.to(device)

    # se voglio fargli fare il giro della pista basta che modifico le iterazioni nell env
    env = PyBulletContinuousEnv()

    #episode_rewards = []
    for i in range(n_test_episodes):
        run_episode(env, agent)
        #episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    # results = dict()
    # results["episode_rewards"] = episode_rewards
    # results["mean"] = np.array(episode_rewards).mean()
    # results["std"] = np.array(episode_rewards).std()
 
    # fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    # fh = open(fname, "w")
    # json.dump(results, fh)
            
    env.close()
    print('... finished')



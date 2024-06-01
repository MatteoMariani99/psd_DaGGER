from __future__ import print_function

from datetime import datetime
import numpy as np
import draw_steering_angle
import json
import torch
import cv2
from model import Model
import math


from environment import PyBulletContinuousEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Torch Device:", device)



def run_episode(env:PyBulletContinuousEnv, agent, max_timesteps=2500):
    
    episode_reward = 0
    step = 0
    env.reset()
    state= env.get_observation()
    

    while True:

        state= env.get_observation()
        colorHSV = env.getCamera_image()
        bird_eye = cv2.resize(colorHSV, (480, 320))
        
        pts = np.array([[0, 0], [480, 0],
                [480, 70], [0, 70]],
               np.int32)
 
        pts = pts.reshape((-1, 1, 2))

        cv2.fillPoly(bird_eye, pts=[pts], color=(0, 0, 0))

            
        steering_wheel = draw_steering_angle.SteeringWheel(bird_eye)
        
        
        #state_torch = torch.from_numpy(state).to(device)
        #state_torch = (state_torch.permute(2,0,1)).unsqueeze(0)

        #state_torch = (state_torch.permute(2,0,1)).unsqueeze(0) # riordino le dimensioni per passarlo a conv2d
        prediction = agent(torch.from_numpy(state[np.newaxis,np.newaxis,...]).type(torch.FloatTensor).to(device))
        # np.newaxis aumenta la dimensione dell'array di 1 (es. se è un array 1D diventa 2D)
        # torch.from_numpy crea un tensore a partire da un'array numpy
        # il modello ritorna le azioni (left/right, up, down)
        #start_time1 = time.time()

        a = prediction.detach().cpu().numpy().flatten()

        result_image = steering_wheel.draw_steering_wheel_on_image(a[0]*180/math.pi,(20,10))
        
        
        vel_image = steering_wheel.update_frame_with_bar(a[1])
        
        #img = cv2.cvtColor(result_image, cv2.COLOR_GRAY2RGB)
        cv2.imshow("Camera2", vel_image)
        cv2.waitKey(1) 
        # per far si che le azioni non sforino vel_max

        print("Action for model: ",a)

        # take action, receive new state & reward
        next_state, reward, done = env.step(a)   
        #episode_reward += reward       
        state = next_state
        step += 1

        if done or step > max_timesteps: 
            break

    return episode_reward


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("-p","--path", required=True, type=str, help="Path to PyTorch model")
    #args = parser.parse_args()

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    #rendering = True                      
    
    n_test_episodes = 15                  # number of episodes to test


    agent = Model()

    agent.load("dagger_test_models/model_optim_params.pth")
    agent.to(device)

    env = PyBulletContinuousEnv()

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')



# provo a fare un rettangolo in basso in modo da farci stare il volante e la parte di velocità
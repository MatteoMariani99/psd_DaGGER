from __future__ import print_function

from datetime import datetime
import numpy as np

import json
import torch

from model import Model


from environment import PyBulletContinuousEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Torch Device:", device)
vel_max = 10

def run_episode(env, agent, max_timesteps=2500):
    
    episode_reward = 0
    step = 0
    env.reset()
    state = env.get_observation()

    while True:
        # preprocessing 
        #gray = np.dot(state[...,:3], [0.2125, 0.7154, 0.0721])[:84,...]
        #pred = agent(torch.from_numpy(gray[np.newaxis, np.newaxis,...]).type(torch.FloatTensor))
        #state = cv2.cvtColor(state, cv2.COLOR_RGB2YUV)
        #state = rgb2yuv(state)
        #cv2.imshow("Camera", state)
        #cv2.waitKey(0) 
        #state_torch = torch.from_numpy(state).to(device)
        #state_torch = (state_torch.permute(2,0,1)).unsqueeze(0)

        #state_torch = (state_torch.permute(2,0,1)).unsqueeze(0) # riordino le dimensioni per passarlo a conv2d
        prediction = agent(torch.from_numpy(state[np.newaxis,np.newaxis,...]).type(torch.FloatTensor).to(device))
        # np.newaxis aumenta la dimensione dell'array di 1 (es. se è un array 1D diventa 2D)
        # torch.from_numpy crea un tensore a partire da un'array numpy
        # il modello ritorna le azioni (left/right, up, down)
        #start_time1 = time.time()

        a = prediction.detach().cpu().numpy().flatten()
        # per far si che le azioni non sforino vel_max
        # if a[1] > 10:
        #     a[1] = 10
        # if a[1] < -10:
        #     a[1] = -10
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

    # TODO: load agent
    #agent = VehicleControlModel()
    agent = Model()
    #print("Loading model {}:".format(args.path))
    agent.load("dagger_test_models/model_{}.pth".format(6))
    agent.to(device)
    # agent.load("models/agent.ckpt")
    #env = gym.make('CarRacing-v0').unwrapped
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

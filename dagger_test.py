from __future__ import print_function
import numpy as np
import pickle
import os
from datetime import datetime
import gzip
import json
from model_new import VehicleControlModel
from model import Model
import train_agent
from utils import *
import torch
import cv2
from environment import PyBulletContinuousEnv
import pybullet as p
from get_track import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using GPU")

# OBSERVATION SPACE: RGB image 96x96x3
# ACTION SPACE: [LEFT/RIGHT, UP, DOWN]
# LEFT/RIGHT: -1, +1
# UP: 0 / +1
# DOWN: 0 / +1


NUM_ITS = 20 # default è 20. Viene utilizzato 1 lap per iteration. Le iteration rappresentano il
# numero di volte che noi stoppiamo l'esperto per salvare i dati e fare il training della rete.
# è un po' come se fosse il numero di episodi.
beta_i  = 0.9 # parametro usato nella policy PI: tale valore verrà modificato tramite la 
# formula curr_beta = beta_i**model_number con model_number che incrementa di 1 ogni volta. 
# Inizialmente avremo 0.9^0, poi 0.9^1 poi 0.9^2 e così via il beta diminuirà esponenzialmente.
# Ciò significa che avremo una probabilità di utilizzare la politica dell'expert che decresce 
# a mano a mano che si procede con il training.
T = 10000 # ogni iteration contiene N passi
vel_max = 15 # velocità massima macchina


s = """  ____    _                         
 |  _ \  / \   __ _  __ _  ___ _ __ 
 | | | |/ _ \ / _` |/ _` |/ _ \ '__|
 | |_| / ___ \ (_| | (_| |  __/ |   
 |____/_/   \_\__, |\__, |\___|_|   
              |___/ |___/           

"""

# funzioni utili per muovere la macchina tramite tastiera
def wait():
    _ = input()


# funzione utile a salvare i dati del percorso utili per il training della rete
def store_data(data, datasets_dir="./data_test"):
    # save data
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'data_dagger.pkl.gzip')
    # apro il file e ci aggancio le nuove immagini
    f = gzip.open(data_file,'wb')
    pickle.dump(data, f)


# funzione che salva i risultati dell'episodio: rewards, rewards mean, rewards std
def save_results(episode_rewards, results_dir="./results_test"):
    # save results
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # save statistics in a dictionary and write them into a .json file
    results = dict()
    results["number_episodes"] = len(episode_rewards)
    results["episode_rewards"] = episode_rewards

    results["mean_all_episodes"] = np.array(episode_rewards).mean()
    results["std_all_episodes"] = np.array(episode_rewards).std()
 
    fname = os.path.join(results_dir, "results_test-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S"))
    fh = open(fname, "w")
    json.dump(results, fh)
    print('... finished')



if __name__ == "__main__":

    print(s)
    print("Welcome to the DAgger Algorithm for CarRacing-v0!")
    print("Drive the car using the arrow keys.")
    print("After every {} timesteps, the game will freeze and train the agent using the collected data.".format(T))
    print("After each training loop, the previously trained network will have more control of the action taken.")
    print("Trained models will be saved in the dagger_models directory.")
    print("Press any key to begin driving!")
    wait()


    if not os.path.exists("dagger_test_models"):
        os.mkdir("dagger_test_models")
 
    # è ciò che estrapoliamo dall'ambiente con l'aggiunta delle azioni e dell'azione terminale di done
    samples = {
        "state": [],
        "next_state": [],
        #"reward": [],
        "action": [],
        "terminal" : [],
    }

    #env = gym.make('CarRacing-v0').unwrapped # utilizzo di unwrapped in modo da agire sull'ambiente
    env = PyBulletContinuousEnv()
    # e modificare i parametri dell'ambiente già costruito di default

    
    episode_rewards = [] # vettore contenente le rewards per ogni episodio
    steps = 0
    #agent = VehicleControlModel() # definisco l'agente dallo script model.py (sarebbe la rete)
    agent = Model()
    agent.save("dagger_test_models/model_0.pth") # salvo il primo modello (vuoto)
    agent.to(device)
    model_number = 0 # inizializzo il numero del modello: aumentandolo varierà beta
    old_model_number = 0 # non serve ai fini pratici
    forward = 10 # 10 m/s

    # qui inizia il vero e proprio algoritmo: num di iterazioni è il numero di episodi che vogliamo
    
    for iteration in range(NUM_ITS):
        #agent = VehicleControlModel() # ridefinisco l'istanza in quanto da questo ciclo non uscirò più
        agent = Model()
        agent.load("dagger_test_models/model_{}.pth".format(model_number)) # carico l'ultimo modello
        agent.to(device)
        curr_beta = beta_i ** model_number # calcolo il coefficiente beta

        # non serve in pratica
        if model_number != old_model_number:
            print("Using model : {}".format(model_number))
            print("Beta value: {}".format(curr_beta))
            old_model_number = model_number

        episode_reward = 0 # inizializzo le reward
        env.reset() # inizializzo l'ambiente e ottengo le misure dello stato che 
        state = env.get_observation() 

        # pi: input to the environment: è la policy utilizzata per collezionare le traiettorie
        # a : expert input

        # Schema
        # 1- uso la politica dell'expert per generare le traiettorie che finiranno nel dataset D
        # 2- train di una policy che meglio imita l'expert su queste traiettorie
        # 3- uso la politica allenata e insieme anche l'expert per collezionare nuove traiettorie
        # e allenare una nuova rete.
        # inizializzato così vuol dire vai dritto (da capire se c'è già una velocità minima
        # oppure è ferma la macchina)
        pi = np.array([0.0, 0.0]).astype('float32') # inizializzo la policy
        a = np.zeros_like(pi) # inizializzo la policy dell'expert

        # ciclo while che permette di eseguire gli step di simulazione fino a che non si raggiungono
        # gli step massimi di simulazioni scelti pari a 4000: oltre questo valore, si entra nell'if
        # si salvano i dati e si allena la rete. Poi si esc3e dal while in modo da aggiornare il modello
        # di agente considerato.
        #start_time = time.time()
        
        
        #while True:
        p.setGravity(0,0,-10)
        
        _,_,centerLine = computeTrack(debug=False)
        total_index = env.computeIndexToStart(centerLine)
        done = False

        for i in total_index:
            goal = centerLine[i][:2] # non prendo la z
            r, yaw_error = env.rect_to_polar_relative(goal)
            #print(f"goal {goal},distance {r}")
            #print("cambio")
            if done:
                break
            while r>0.5:
                # state è l'immagine birdeye
                #cv2.imshow("Camera", state)
                #cv2.waitKey(1)

                r, yaw_error = env.rect_to_polar_relative(goal)
                #print(f"goal {goal},distance {r} ,yaw_error{yaw_error}")
                vel_ang = env.p_control(yaw_error)

                if 199<i<210:
                    forward = 2
                        
                # la riga 161 del dagger.py deve essere sostituita con questa
                a = np.array([vel_ang, forward]).astype('float32')
                
                # passo di simulazione che restituisce lo stato, reward e il flag done
                # nel nostro caso l'ambiente va creato from scratch e va implementata la logica
                # per acquisire le immagini e la funzione di ricompense (anche se quest'ultima non
                # è necessaria)
                # next_state è già bird-eye in formato canny filter
                next_state, rewards, done = env.step(pi) # next_state già in formato YUV

                # preprocess image and find prediction ->  policy(state)
                # passo alla rete lo stato (image) e ottengo le predizioni è come se fosse:
                # action = PPO(state)
                
                #next_state_torch = torch.from_numpy(next_state).to(device)
                #next_state_torch = (next_state_torch.permute(2,0,1)).unsqueeze(0) # riordino le dimensioni per passarlo a conv2d

                # np.newaxis aumenta la dimensione dell'array di 1 (es. se è un array 1D diventa 2D)
                # torch.from_numpy crea un tensore a partire da un'array numpy
                # il modello ritorna le azioni (left/right, up, down)
                #start_time1 = time.time()
                
                prediction = agent(torch.from_numpy(next_state[np.newaxis,np.newaxis,...]).type(torch.FloatTensor).to(device))
                
                #prediction = agent(next_state_torch.type(torch.FloatTensor).to(device))
                #print("--- %s seconds ---" % (time.time() - start_time1))
                # calculate linear combination of expert and network policy
                # pi è la policy: inizialmente ci sarà solo a ovvero le azioni dell'esperto: a mano a mano
                # tramite il coefficiente beta, si avrà anche un peso derivante dalle azioni calcolate dalla
                # rete
                pi = curr_beta * a + (1 - curr_beta) * prediction.detach().cpu().numpy().flatten()
                #print("Policy pi: ",pi)
                print("Policy pred: ",prediction.detach().cpu().numpy().flatten())
                print("Policy a: ",a)
                #episode_reward += r

                samples["state"].append(state)            # state has shape (96, 96, 3)
                samples["action"].append(np.array(a))     # action has shape (1, 3), STORE THE EXPERT ACTION
                samples["next_state"].append(next_state)
                #samples["reward"].append(rewards)
                samples["terminal"].append(done)
                
                state = next_state
                steps += 1

                # dopo T = N step avviene il train della rete
                if steps % T == 0:
                    env.stoppingCar()
                    print('... saving data')
                    store_data(samples, "./data_test")
                    #print("fine: ",episode_rewards)
                    #save_results(episode_rewards, "./results_test")
                    # X_train sono le immagini
                    # y_train sono le label
                    # viene richiamata la funzione train_agent.read_data() che permette di leggere
                    # i dati pickle e scomporli in train e validation set
                    X_train, y_train, X_valid, y_valid = train_agent.read_data("./data_test", "data_dagger.pkl.gzip")
                    # funzione di preprocessing per andare a trasformare l'immagine da colori a scala di grigi
                    #X_train, y_train, X_valid, y_valid = train_agent.preprocessing(X_train, y_train, X_valid, y_valid, history_length=1)
                    print(X_train.shape)
                    train_agent.train_model(X_train, y_train, X_valid, y_valid, "dagger_test_models/model_{}.pth".format(model_number+1), num_epochs=10)
                    model_number += 1
                    print("Training complete. Press return to continue to the next iteration")
                    wait()
                    break

            
            # done esce da env.step in cui però gli step massimi sono 1000
            # in questo modo facciamo steps%T in modo che sia True solamente dopo 4 richiami di 
            # env.step ovvero 4000 step massimi

                if done: 
                    break
                    
        #print("--- %s seconds ---" % (time.time() - start_time))
        
        #episode_rewards.append(episode_reward)

    env.close()

      

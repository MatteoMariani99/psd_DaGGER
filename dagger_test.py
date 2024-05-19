from __future__ import print_function
import numpy as np
import pickle
import os
from datetime import datetime
import gzip
import json
from model_new import VehicleControlModel
import train_agent
from utils import *
import torch
import cv2
from environment import PyBulletContinuousEnv
import pybullet as p
import time



# OBSERVATION SPACE: RGB image 96x96x3
# ACTION SPACE: [LEFT/RIGHT, UP, DOWN]
# LEFT/RIGHT: -1, +1
# UP: 0 / +1
# DOWN: 0 / +1


NUM_ITS = 10 # default è 20. Viene utilizzato 1 lap per iteration. Le iteration rappresentano il
# numero di volte che noi stoppiamo l'esperto per salvare i dati e fare il training della rete.
# è un po' come se fosse il numero di episodi.
beta_i  = 0.9 # parametro usato nella policy PI: tale valore verrà modificato tramite la 
# formula curr_beta = beta_i**model_number con model_number che incrementa di 1 ogni volta. 
# Inizialmente avremo 0.9^0, poi 0.9^1 poi 0.9^2 e così via il beta diminuirà esponenzialmente.
# Ciò significa che avremo una probabilità di utilizzare la politica dell'expert che decresce 
# a mano a mano che si procede con il training.
T = 2000 # ogni iteration contiene N passi
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
 
    # è ciò che strapoliamo dall'ambiente con l'aggiunta delle azioni e dell'azione terminale di done
    samples = {
        "state": [],
        "next_state": [],
        "reward": [],
        "action": [],
        "terminal" : [],
    }

    #env = gym.make('CarRacing-v0').unwrapped # utilizzo di unwrapped in modo da agire sull'ambiente
    env = PyBulletContinuousEnv()
    # e modificare i parametri dell'ambiente già costruito di default

    #env.reset() # inizializzo l'ambiente
    #env.viewer.window.on_key_press = key_press
    #env.viewer.window.on_key_release = key_release
    
    episode_rewards = [] # vettore contenente le rewards per ogni episodio
    steps = 0
    agent = VehicleControlModel() # definisco l'agente dallo script model.py (sarebbe la rete)
    agent.save("dagger_test_models/model_0.pth") # salvo il primo modello (vuoto)
    model_number = 0 # inizializzo il numero del modello: aumentandolo varierà beta
    old_model_number = 0 # non serve ai fini pratici


    # qui inizia il vero e proprio algoritmo: num di iterazioni è il numero di episodi che vogliamo
    
    for iteration in range(NUM_ITS):
        agent = VehicleControlModel() # ridefinisco l'istanza in quanto da questo ciclo non uscirò più
        agent.load("dagger_test_models/model_{}.pth".format(model_number)) # carico l'ultimo modello
        curr_beta = beta_i ** model_number # calcolo il coefficiente beta

        # non serve in pratica
        if model_number != old_model_number:
            print("Using model : {}".format(model_number))
            print("Beta value: {}".format(curr_beta))
            old_model_number = model_number

        episode_reward = 0 # inizializzo le reward
        env.reset() # inizializzo l'ambiente e ottengo le misure dello stato che 
        state = env.get_observation() 
        # sono immagini 96x96x3
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


        forward = 0
        turn = 0
        #backward = 0

        # ciclo while che permette di eseguire gli step di simulazione fino a che non si raggiungono
        # gli step massimi di simulazioni scelti pari a 4000: oltre questo valore, si entra nell'if
        # si salvano i dati e si allena la rete. Poi si esc3e dal while in modo da aggiornare il modello
        # di agente considerato.
        #start_time = time.time()
        
  
        while True:
            
            #a = env.getAction_expert()
            p.setGravity(0,0,-10)
            time.sleep(1./240.)
            keys = p.getKeyboardEvents()

            # comandi da tastiera per l'expert
            for k,v in keys.items():

                    if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                            turn = -0.5
                    if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_RELEASED)):
                            turn = 0
                    if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                            turn = 0.5
                    if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_RELEASED)):
                            turn = 0
                    if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                            forward=15
                    if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_RELEASED)):
                            forward=0
                    if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                            forward=-15
                    if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_RELEASED)):
                            forward=0

            
            # la riga 161 del dagger.py deve essere sostituita con questa
            a = np.array([turn, forward]).astype('float32')
            # passo di simulazione che restituisce lo stato, reward e il flag done
            # nel nostro caso l'ambiente va creato from scratch e va implementata la logica
            # per acquisire le immagini e la funzione di ricompense (anche se quest'ultima non
            # è necessaria)
            next_state, r, done = env.step(pi) # next_state già in formato YUV
            #cv2.imshow("Camera", state)
            #cv2.waitKey(0) 
            #env.visualization_image()

            #cv2.imshow('Original', next_state) 
            #cv2.waitKey(0) 
            # gray_image = cv2.cvtColor(next_state, cv2.COLOR_BGR2GRAY) 
            #cv2.destroyAllWindows()

            # preprocess image and find prediction ->  policy(state)
            # passo alla rete lo stato (image) e ottengo le predizioni è come se fosse:
            # action = PPO(state)

            # Le immagini vengono ora convertite in grayscale in quanto richieste dal modello come
            # channel di input

            # next_state[...,:3]: l'ultima dimensione corrisponde ai canali colore (es. RGB)
            # next_state è un multi-dimensional array che rappresenta un'immagine
  
            # [0.2125, 0.7154, 0.0721]: sono i pesi usati per convertire RGB to grayscale
            # pesa di più il canale del verde (la vista umana è più sensbile alla luce verde)

            # np.dot(...): fa la moltiplicazione matriciale 

            # [:84,...]: permette di definire l'altezza pari a 84. I puntini servono a mantenere inalterate
            # le altre dimensioni
            # 3 modi
            #gray = cv2.cvtColor(next_state, cv2.COLOR_RGB2GRAY)
            #gray = gray[:84, :]
            # oppure
            #gray = np.dot(next_state[...,:3], [0.2989, 0.5870, 0.1140])
            # oppure
            # gray = cv2.transform(next_state, np.array([[0.2125, 0.7154, 0.0721]]))
            # Crop the image
            #gray = gray[:84, :]
            
            # capire quale va usato dei due (due risultati diversi)
            # quello di opencv difficile da utilizzare in train_agent.py
            #next_state = rgb2yuv(next_state)
            next_state = cv2.cvtColor(next_state,cv2.COLOR_RGB2YUV)
            #cv2.imshow("Camera", next_state)
            #cv2.waitKey(0) 
            
            next_state_torch = torch.from_numpy(next_state)
            next_state_torch = (next_state_torch.permute(2,0,1)).unsqueeze(0) # riordino le dimensioni per passarlo a conv2d


            #print(next_state.shape)
            #print(state.shape)
            
            #image = np.transpose(image, (2, 0, 1)) # per avere l'ordine giusto delle dimensioni
            
            #print(image.shape)
            # np.newaxis aumenta la dimensione dell'array di 1 (es. se è un array 1D diventa 2D)
            # torch.from_numpy crea un tensore a partire da un'array numpy
            # il modello ritorna le azioni (left/right, up, down)
            #start_time1 = time.time()

            
            #prediction = agent(torch.from_numpy(gray[np.newaxis,np.newaxis,...]).type(torch.FloatTensor))
            
            prediction = agent(next_state_torch.type(torch.FloatTensor))
            #print("--- %s seconds ---" % (time.time() - start_time1))
            # calculate linear combination of expert and network policy
            # pi è la policy: inizialmente ci sarà solo a ovvero le azioni dell'esperto: a mano a mano
            # tramite il coefficiente beta, si avrà anche un peso derivante dalle azioni calcolate dalla
            # rete
            pi = curr_beta * a + (1 - curr_beta) * prediction.detach().numpy().flatten()
            #pi = a
            print("Policy pi: ",pi)
            print("Policy a: ",a)
            episode_reward += r
            #print("Episode: ",episode_reward)

            samples["state"].append(state)            # state has shape (96, 96, 3)
            samples["action"].append(np.array(a))     # action has shape (1, 3), STORE THE EXPERT ACTION
            samples["next_state"].append(next_state)
            samples["reward"].append(r)
            samples["terminal"].append(done)
            
            state = next_state
            steps += 1

            # dopo T = N step avviene il train della rete

            if steps % T == 0:
                env.stoppingCar()
                print('... saving data')
                store_data(samples, "./data_test")
                #print("fine: ",episode_rewards)
                save_results(episode_rewards, "./results_test")
                # X_train sono le immagini
                # y_train sono le label
                # viene richiamata la funzione train_agent.read_data() che permette di leggere
                # i dati pickle e scomporli in train e validation set
                X_train, y_train, X_valid, y_valid = train_agent.read_data("./data_test", "data_dagger.pkl.gzip")
                # funzione di preprocessing per andare a trasformare l'immagine da colori a scala di grigi
                X_train, y_train, X_valid, y_valid = train_agent.preprocessing(X_train, y_train, X_valid, y_valid, history_length=1)
                #print(X_train.shape)
                train_agent.train_model(X_train, y_train, X_valid, y_valid, "dagger_test_models/model_{}.pth".format(model_number+1), num_epochs=10)
                model_number += 1
                print("Training complete. Press return to continue to the next iteration")
                wait()
                break

            #env.render()
            # done esce da env.step in cui però gli step massimi sono 1000
            # in questo modo facciamo steps%T in modo che sia True solamente dopo 4 richiami di 
            # env.step ovvero 4000 step massimi

            if done: 
                break
        #print("--- %s seconds ---" % (time.time() - start_time))
        
        episode_rewards.append(episode_reward)

    env.close()

      

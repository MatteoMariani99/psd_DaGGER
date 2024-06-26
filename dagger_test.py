import cv2
import numpy as np
import pickle
import os
from datetime import datetime
import gzip
import json
import train_agent
import torch
from model import Model
import time
from environment_cones import PyBulletContinuousEnv
from get_track import *
from get_cones import *

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using GPU")

# OBSERVATION SPACE: bird eye image 96x96x1
# ACTION SPACE: [LEFT/RIGHT, UP/DOWN]
# LEFT/RIGHT: -1, +1
# UP/DOWN: -1 / +1



NUM_ITS = 41 # default è 20. Viene utilizzato 1 lap per iteration. Le iteration rappresentano il
# numero di volte che noi stoppiamo l'esperto per salvare i dati e fare il training della rete.
# è un po' come se fosse il numero di episodi.
beta_i  = 0.9 # parametro usato nella policy PI: tale valore verrà modificato tramite la 
# formula curr_beta = beta_i**model_number con model_number che incrementa di 1 ogni volta. 
# Inizialmente avremo 0.9^0, poi 0.9^1 poi 0.9^2 e così via il beta diminuirà esponenzialmente.
# Ciò significa che avremo una probabilità di utilizzare la politica dell'expert che decresce 
# a mano a mano che si procede con il training.
T = 4000 # ogni iteration contiene N passi

s = """  ____    _                         
 |  _ \  / \   __ _  __ _  ___ _ __ 
 | | | |/ _ \ / _` |/ _` |/ _ \ '__|
 | |_| / ___ \ (_| | (_| |  __/ |   
 |____/_/   \_\__, |\__, |\___|_|   
              |___/ |___/           

"""

# funzione che attende l'input da parte dell'utente
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
    print("Welcome to the DAgger Algorithm")
    print("After every {} timesteps, the simulation will freeze and train the agent using the collected data.".format(T))
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
        "action": [],
        "terminal" : [],
    }

    # istanza dell'ambiente
    env = PyBulletContinuousEnv()
    
    # istanza del modello
    agent = Model()
    agent.save("dagger_test_models/model_0.pth") # salvo il primo modello (vuoto)
    agent.to(device) # passo alla gpu
    
    episode_rewards = [] # vettore contenente le rewards per ogni episodio
    steps = 0
    model_number = 0 # inizializzo il numero del modello: aumentandolo varierà beta
    old_model_number = 0 # non serve ai fini pratici
    

    
    # qui inizia il vero e proprio algoritmo: num di iterazioni è il numero di episodi che vogliamo
    for iteration in range(NUM_ITS):
        
        # reinizializzo il modello in quanto da questo ciclo poi non uscirò più
        agent = Model()
        agent.load("dagger_test_models/model_{}.pth".format(model_number)) # carico l'ultimo modello
        agent.to(device)
        curr_beta = beta_i ** model_number # calcolo il coefficiente beta

        # stampo a video il numero di modello utilizzato e il coefficiente beta
        if model_number != old_model_number:
            print("Using model : {}".format(model_number))
            print("Beta value: {}".format(curr_beta))
            old_model_number = model_number

        episode_reward = 0 # inizializzo le reward
        env.reset() # inizializzo l'ambiente
        
        
        state = env.get_observation() # ottengo le osservazioni ovvero le immagini 84x96x1 che andranno poi passate alla rete
        
        # pi: input to the environment: è la policy utilizzata per collezionare le traiettorie
        # a : expert input

        # Schema
        # 1- uso la politica dell'expert per generare le traiettorie che finiranno nel dataset D
        # 2- train di una policy che meglio imita l'expert su queste traiettorie
        # 3- uso la politica allenata e insieme anche l'expert per collezionare nuove traiettorie
        # e allenare una nuova rete.
        # inizializzato ferma la macchina
        pi = np.array([0.0, 0.0]).astype('float32') # inizializzo la policy
        a = np.zeros_like(pi) # inizializzo la policy dell'expert
        
 
        # ottengo i punti centrali della strada
        #_,_,centerLine = computeTrack(debug=False)
        _,_,centerLine = getCones()
        
        # assemblo la lista dei punti a partire dal punto che mi serve
        # es. se parto dalla posizione (10,10) posso trovarmi a metà lista: in questo modo faccio si che 
        # la posizione (10,10) sia quella in testa alla lista in quanto sarà quella da cui parto
        # e di resto andranno tutti gli altri punti
        total_index = env.computeIndexToStart(centerLine)
        done = False

        # per ogni indice nella lista seguo il punto tramite il controllore P
        for i in total_index:
            goal = centerLine[i][:2] # non prendo la z
            r, yaw_error = env.rect_to_polar_relative(goal)

            # viene messo sia qui che sotto in modo da uscire prima dal ciclo for e poi dal while
            # done = True quando abbiamo completato il numero di step in env
            if done:
                break
            
            # quando la distanza verso il punto diventa minore di 0.5 passo al punto successivo
            while r>0.5:

                # calcolo le coordinate polari dalla macchina al punto che devo raggiungere ottenendo in uscita distanza (m) e angolo (rad)
                r, yaw_error = env.rect_to_polar_relative(goal)
                
                # chiamo in causa il controllore P che dato l'errore sull'angolo di orientazione della macchina rispetto al goal
                # mappa l'output in velocità angolare e lineare da comandare al veicolo
                vel_ang,vel_lin = env.p_control(yaw_error)
                        
                # definisco le azioni dell'esperto come vel_ang in uscita al controllore e forward costante a 10
                a = np.array([vel_ang, vel_lin]).astype('float32')
                
                # passo di simulazione che restituisce lo stato, reward e il flag done
                # nel nostro caso l'ambiente va creato from scratch e va implementata la logica
                # per acquisire le immagini e la funzione di ricompense (anche se quest'ultima non
                # è necessaria)
                #start1 = time.time()
                next_state, rewards, done = env.step(pi) 
                cv2.imshow("YOLOv8 Tracking", next_state)
                # Display the annotated frame
                #cv2.imshow("YOLOv8 detect", annotated_frame)
                #cv2.imshow('IMAGE', img)
                cv2.waitKey(1)
                #print("-----secondi-----", time.time()-start1)
                

                # preprocess image and find prediction ->  policy(state)
                # passo alla rete lo stato (image) e ottengo le predizioni.
                
                # np.newaxis aumenta la dimensione dell'array di 1 (es. se è un array 1D diventa 2D)
                # torch.from_numpy crea un tensore a partire da un'array numpy
                # il modello ritorna le azioni (left/right, up/down)
                prediction = agent(torch.from_numpy(next_state[np.newaxis,np.newaxis,...]).type(torch.FloatTensor).to(device))
                
                # calculate linear combination of expert and network policy
                # pi è la policy: inizialmente ci sarà solo a ovvero le azioni dell'esperto: a mano a mano
                # tramite il coefficiente beta, si avrà anche un peso derivante dalle azioni calcolate dalla
                # rete
                pi = curr_beta * a + (1 - curr_beta) * prediction.detach().cpu().numpy().flatten()

                print("Policy pred: ",prediction.detach().cpu().numpy().flatten())
                print("Policy a: ",a)
                
                if steps%2==0:
                    samples["state"].append(state)            # state has shape (96, 96, 1)
                    samples["action"].append(np.array(a))     # action has shape (1, 3), STORE THE EXPERT ACTION
                    samples["next_state"].append(next_state)
                    samples["terminal"].append(done)
                
                state = next_state
                steps += 1
                
                # dopo T = N step avviene il train della rete
                if steps % T == 0:
                    
                    env.stoppingCar()
                    
                    print('... saving data')
                    store_data(samples, "./data_test")

                    # X_train sono le immagini
                    # y_train sono le label
                    # viene richiamata la funzione train_agent.read_data() che permette di leggere
                    # i dati pickle e scomporli in train e validation set
                    X_train, y_train, X_valid, y_valid = train_agent.read_data("./data_test", "data_dagger.pkl.gzip")

                    print("Immagini per il training: ",X_train.shape)
                    
                    #? USANDO PYTORCH LIGHTNING
                    # definisco il modello pytorch lightning
                    # light_model = LitAgentTrain(learning_rate=1e-3, batch_size=16, dataset_train=list(zip(X_train, y_train)), dataset_val=list(zip(X_valid, y_valid)))

                    # logger = TensorBoardLogger("tb_logs", name="my_model")
                    # trainer = pl.Trainer(max_epochs=10, 
                    #                 accelerator='gpu', 
                    #                 logger=logger)
                    
                    # # training modello con pytorch lightning
                    # trainer.fit(model=light_model)
                    
                    # agent.save("dagger_test_models/model_{}.pth".format(model_number+1))
                    
                    
                    train_loss = train_agent.train_model(X_train, y_train, "dagger_test_models/model_{}.pth".format(model_number+1), num_epochs=10)
                    writer.add_scalar("Loss/train", train_loss, iteration)
                    
                    model_number += 1
                    val_loss = train_agent.validate_model(X_valid,y_valid, agent)
                    writer.add_scalar("Loss/val", val_loss, iteration)
                    writer.flush()
                    print("Training and validation complete. Press return to continue to the next iteration")
                    wait()
                    break

            
            # done esce da env.step in cui però gli step massimi sono 1000
            # in questo modo facciamo steps%T in modo che sia True solamente dopo 4 richiami di 
            # env.step ovvero 4000 step massimi

                if done: 
                    break

    env.close()

      

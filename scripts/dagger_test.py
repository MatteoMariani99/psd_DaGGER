import argparse
import cv2
import numpy as np
import pickle
import os
import gzip
import train_agent
import torch
from torch.utils.tensorboard import SummaryWriter

#? Import ambienti
from environment_cones import ConesEnv
from environment_road import RoadEnv
from get_track import *
from get_cones import *
from model import Model

# utile al salvataggio dei dati su tensorboard
writer = SummaryWriter()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")

# OBSERVATION SPACE: bird eye image 84x96x1
# ACTION SPACE: [LEFT/RIGHT, UP/DOWN]
# LEFT/RIGHT: -1, +1
# DOWN/UP: -1 / +1



#NUM_ITS = 49 # Le iteration rappresentano il numero di episodi per il quale vogliamo eseguire la simulazione.
beta_i  = 0.9 # parametro usato nella policy PI: tale valore verrà modificato tramite la 
# formula curr_beta = beta_i**model_number con model_number che incrementa di 1 ogni volta. 
# Inizialmente avremo 0.9^0, poi 0.9^1 poi 0.9^2 e così via il beta diminuirà esponenzialmente.
# Ciò significa che avremo una probabilità di utilizzare la politica dell'expert che decresce 
# a mano a mano che si procede con il training.
T = 4000 # numero di passi utili prima di eseguire il training (step = 1000 e quindi ogni 4 volte eseguiamo il training)

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



def store_data(data, datasets_dir="./data_test"):
    """
    Funzione utile a salvare i dati del percorso per il training della rete
    
    Parameters:
        - data: dati contenenti immagini e label della simulazione
        - dataset_dir: nome della directory in cui salvare i dati
        
    Returns:
        - loss: calcolo della loss 
    """
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'data_dagger.pkl.gzip')
    
    # apertura file e scrittura dati
    f = gzip.open(data_file,'wb')
    pickle.dump(data, f)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description=
                                     "DAGGER\n"
                                     "Default: --cones = False\t specifica il tracciato\n"
                                     "Default: --num_its = 49\t\t specifica il numero di iterazioni",
                                     formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('--cones', action='store_true',help='tipologia di tracciato')
    parser.add_argument('--num_its',type=int, default=49,help="numero di iterazioni/episodi")
    args = parser.parse_args()
    
    NUM_ITS = args.num_its

    print(s)
    print("Welcome to the DAgger Algorithm")
    print("After every {} timesteps, the simulation will freeze and train the agent using the collected data.".format(T))
    print("After each training loop, the previously trained network will have more control of the action taken.")
    print("Trained models will be saved in the dagger_models directory.")
    print("Press any key to begin driving!")
    wait()



    if not os.path.exists("dagger_models"):
        os.mkdir("dagger_models")
 
    # rappresenta i dati che salveremo successivamente:
    # state: immagini in birdeye
    # action: sono le azioni che applichiamo sull'ambiente
    # terminal: quando si concludono gli step
    samples = {
        "state": [],
        "action": [],
        "terminal" : [],
    }

    # scelta ambiente
    cones = args.cones
    if cones:
    # istanza dell'ambiente
        env = ConesEnv()
    else:
        env = RoadEnv()
    
    # istanza del modello
    agent = Model()
    agent.save("dagger_models/model_30.pth") # salvo il primo modello (vuoto)
    agent.to(device) # passo alla gpu
    
    steps = 0 # inizializzo il numero di step da eseguire per una simulazione
    model_number = 0 # inizializzo il numero del modello: aumentandolo varierà beta
    old_model_number = 0 
    

    
    # qui inizia il vero e proprio algoritmo: NUM_ITS è il numero di episodi che vogliamo
    for iteration in range(NUM_ITS):
        
        # reinizializzazione del modello in quanto da questo ciclo non s uscirà
        agent = Model()
        agent.load("dagger_models/model_{}.pth".format(model_number), device) # carico l'ultimo modello
        agent.to(device)
        curr_beta = beta_i ** model_number # calcolo il coefficiente beta

        # stampo a video il numero di modello utilizzato e il coefficiente beta
        if model_number != old_model_number:
            print("Using model : {}".format(model_number))
            print("Beta value: {}".format(curr_beta))
            old_model_number = model_number

        env.reset() # inizializzo l'ambiente
        
        state = env.get_observation() # ottengo le osservazioni ovvero le immagini 84x96x1 che andranno poi passate alla rete
        
        
        # Schema
        # 1- uso la politica dell'expert per generare le traiettorie che finiranno nel dataset 
        # 2- train di una policy che meglio imita l'expert su queste traiettorie
        # 3- uso la politica allenata insieme all'expert per collezionare nuove traiettorie
        # e allenare una nuova rete.
        
        # pi: è la policy utilizzata per collezionare le traiettorie (expert + rete)
        # a : expert input
        pi = np.array([0.0, 0.0]).astype('float32') # inizializzo la policy
        a = np.zeros_like(pi) # inizializzo la policy dell'expert
        
 
        # ottengo i punti centrali della strada
        if cones:
            _,_,centerLine = getCones(env.track_number)
        else:
            _,_,centerLine = computeTrack(debug=False)

        
        # assemblo la lista dei punti a partire dal punto che mi serve
        # es. se parto dalla posizione (10,10) posso trovarmi a metà lista: in questo modo faccio si che 
        # la posizione (10,10) sia quella in testa alla lista in quanto sarà quella da cui parto
        # e di resto andranno tutti gli altri punti
        total_index = env.computeIndexToStart(centerLine)
        done = False

        # per ogni indice nella lista seguo il punto tramite il controllore P
        for i in total_index:
            goal = centerLine[i][:2] # non prendo la z
            
            # calcolo le coordinate polari dalla macchina al punto che devo raggiungere ottenendo in uscita distanza (m) e angolo (rad)
            r, yaw_error = env.rect_to_polar_relative(goal)

            # viene messo sia qui che sotto in modo da uscire prima dal ciclo while e poi dal ciclo for
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
                        
                # definisco le azioni dell'esperto come vel_ang e vel_lin in uscita al controllore 
                a = np.array([vel_ang, vel_lin]).astype('float32')
                
                # passo di simulazione che restituisce lo stato (immagini bird eye) e il flag done
                next_state, _, done = env.step(pi) 
                cv2.imshow("Bird eye", next_state)
                cv2.waitKey(1)
                
                # passo alla rete lo stato (image) e ottengo le predizioni.
                # np.newaxis aumenta la dimensione dell'array di 1 (es. se è un array 1D diventa 2D)
                # torch.from_numpy crea un tensore a partire da un'array numpy
                # il modello ritorna le azioni (left/right, up/down)
                prediction = agent(torch.from_numpy(next_state[np.newaxis,np.newaxis,...]).type(torch.FloatTensor).to(device))
                
                
                # pi è la policy: inizialmente ci sarà solo a ovvero le azioni dell'esperto: a mano a mano
                # tramite il coefficiente beta, si avrà anche un peso derivante dalle azioni calcolate dalla
                # rete
                pi = curr_beta * a + (1 - curr_beta) * prediction.detach().cpu().numpy().flatten()

                samples["state"].append(state)            # state has shape (96, 96, 1)
                samples["action"].append(np.array(a))     # action has shape (1, 3), STORE THE EXPERT ACTION
                samples["terminal"].append(done)
                
                state = next_state
                steps += 1
                
                # dopo step = T = 4000 avviene il training della rete
                if steps % T == 0:
                    
                    env.stoppingCar() # si ferma la macchina
                    
                    print('... saving data')
                    store_data(samples, "./data_test")

                    # X_train sono le immagini
                    # y_train sono le label
                    # viene richiamata la funzione train_agent.read_data() che permette di leggere
                    # i dati dal formato pickle e scomporli in train e validation set
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
                    
                    # agent.save("dagger_models/model_{}.pth".format(model_number+1))
                    
                    # loop di training per il calcolo della loss
                    train_loss = train_agent.train_model(X_train, y_train, "dagger_models/model_{}.pth".format(model_number+1), num_epochs=15)
                    writer.add_scalar("Loss/train", train_loss, iteration) # caricamento su tensorboard 
                    
                    model_number += 1
                    
                    # loop di validazione 
                    val_loss = train_agent.validate_model(X_valid,y_valid, agent)
                    writer.add_scalar("Loss/val", val_loss, iteration) # caricamento su tensorboard 
                    writer.flush()
                    print("Training and validation complete. Press return to continue to the next iteration")
                    wait()
                    break

            
                # se ho raggiunto il numero massimo di step concludo la simulazione e passo alla successiva
                if done: 
                    break

    env.close()

      

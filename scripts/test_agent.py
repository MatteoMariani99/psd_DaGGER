import numpy as np
import draw_steering_angle
import torch
import cv2
import math
from model import Model
import argparse

#? Import ambienti
from environment_cones import ConesEnv
from environment_road import RoadEnv
from get_track import computeTrack
from get_cones import getCones


#* Scelta del device di utilizzo: se è presente una GPU viene utilizzata
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Torch Device:", device)



# funzione usata per mostrare volante e velocità
def draw_steer_speed(state, a):
    color_rgb = env.getCamera_image()
    bird_eye = cv2.resize(color_rgb, (480, 320))
    #punti per volante e barra verticale
    pts = (np.array([[0, 0], [480, 0],
            [480, 70], [0, 70]],
            np.int32)).reshape((-1, 1, 2))
    cv2.fillPoly(bird_eye, pts=[pts], color=(0, 0, 0))

    steering_wheel = draw_steering_angle.SteeringWheel(bird_eye)
    
    # disegno il volante per lo sterzo 
    steering_wheel.draw_steering_wheel_on_image(a[0]*180/math.pi,(20,10))
    # aggiungo la barra verticale per la velocità
    vel_image = steering_wheel.update_frame_with_bar(a[1])
    
    text = " rad/s"
    full_text = f"{str(round(a[0],3))}{text}" 
    text1 = " m/s"
    full_text1 = f"{str(round(a[1],2))}{text1}" 
    
    # Display del testo a video completi
    cv2.putText(vel_image, full_text, (90,40), cv2.FONT_HERSHEY_SIMPLEX,  
                     0.6, (255,255,255), 1, cv2.LINE_AA) 
    cv2.putText(vel_image, full_text1, (360,40), cv2.FONT_HERSHEY_SIMPLEX,  
                     0.6, (255,255,255), 1, cv2.LINE_AA) 
    
    image_obs = state
    image_obs = cv2.resize(state, (480, 320))
    image_obs = cv2.cvtColor(image_obs, cv2.COLOR_GRAY2RGB)

    cv2.imshow("Camera2",  cv2.vconcat([vel_image, image_obs]))
    cv2.waitKey(1) 



def run_episode_save_pose(max_timesteps=2000):
    
    step = 0
    env.reset() # inizializzazione l'ambiente
    state= env.get_observation() # ottenimento delle osservazioni
    file_path = "trajectory/poses/nome_file.txt"
    controller = True # se si vuole salvare la traiettoria del controllore
    
    while True:
        with open(file_path,"a") as f:
            # ottengo i punti centrali della strada
            if controller:
                if cones:
                    _,_,centerLine = getCones(env.track_number)
                    #centerLine = centerLine[::-1] # peer il tracciato 6   
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
                    pos, _ = env.getCarPosition()
                    print(goal, pos)
                    
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
                        next_state, _, done = env.step(a)
                        
                        pos, _ = env.getCarPosition()
                
                        f.write(f"{pos[0]:.4f}, {pos[1]:.4f}\n")
            else:
                # np.newaxis aumenta la dimensione dell'array di 1 (es. se è un array 1D diventa 2D)
                # torch.from_numpy crea un tensore a partire da un'array numpy
                # il modello ritorna le azioni (left/right, up/down)
                prediction = agent(torch.from_numpy(state[np.newaxis,np.newaxis,...]).type(torch.FloatTensor).to(device))
                a = prediction.detach().cpu().numpy().flatten()
                    
                if not cones:
                    draw_steer_speed(state,a)
                else:
                    # inizializzo un'immagine vuota per stampare in seguito i valori di velocità e sterzo
                    state_image = np.zeros((320,480), dtype=np.uint8)

                    text = " rad/s"
                    full_text = f"{str(round(a[0],3))}{text}" 
                    text1 = " m/s"
                    full_text1 = f"{str(round(a[1],2))}{text1}" 
                    
                    # Display dei valori a video
                    cv2.putText(state_image, "Steer: ", (150,100), cv2.FONT_HERSHEY_SIMPLEX,  
                                    0.6, (255,255,255), 1, cv2.LINE_AA)
                    cv2.putText(state_image, full_text, (230,100), cv2.FONT_HERSHEY_SIMPLEX,  
                                    0.6, (255,255,255), 1, cv2.LINE_AA) 
                    cv2.putText(state_image, "Speed: ", (150,250), cv2.FONT_HERSHEY_SIMPLEX,  
                                    0.6, (255,255,255), 1, cv2.LINE_AA)
                    cv2.putText(state_image, full_text1, (230,250), cv2.FONT_HERSHEY_SIMPLEX,  
                                    0.6, (255,255,255), 1, cv2.LINE_AA) 
                    cv2.imshow("Camera", cv2.vconcat([cv2.resize(state,(480,320)), state_image]))
                    k = cv2.waitKey(1)
                    if k==ord('p'):
                        cv2.waitKey(0)
            
                # data l'azione sopra ottenuta, si ottiene la nuova immagine
                next_state, _, done = env.step(a)
                
                # posa del veicolo
                pos, _ = env.getCarPosition()
                
                # scrittura su file
                f.write(f"{pos[0]:.4f}, {pos[1]:.4f}\n")
                
                
            state = next_state
            step += 1

            # si interrompe quando si raggiunge il numero massimo di step
            if done or step > max_timesteps: 
                break
            
            

def run_episode(max_timesteps=2000):
    
    step = 0
    env.reset() # inizializzazione l'ambiente
    state= env.get_observation() # ottenimento delle osservazioni
    
    
    while True:

        # np.newaxis aumenta la dimensione dell'array di 1 (es. se è un array 1D diventa 2D)
        # torch.from_numpy crea un tensore a partire da un'array numpy
        # il modello ritorna le azioni (left/right, up/down)
        prediction = agent(torch.from_numpy(state[np.newaxis,np.newaxis,...]).type(torch.FloatTensor).to(device))
        a = prediction.detach().cpu().numpy().flatten()
        
        
        # interfaccia diversi per coni e road
        if not cones:
            draw_steer_speed(state,a)
        else:
            # inizializzo un'immagine vuota per stampare in seguito i valori di velocità e sterzo
            state_image = np.zeros((320,480), dtype=np.uint8)

            text = " rad/s"
            full_text = f"{str(round(a[0],3))}{text}" 
            text1 = " m/s"
            full_text1 = f"{str(round(a[1],2))}{text1}" 
            
            # Display dei valori a video
            cv2.putText(state_image, "Steer: ", (150,100), cv2.FONT_HERSHEY_SIMPLEX,  
                              0.6, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(state_image, full_text, (230,100), cv2.FONT_HERSHEY_SIMPLEX,  
                              0.6, (255,255,255), 1, cv2.LINE_AA) 
            cv2.putText(state_image, "Speed: ", (150,250), cv2.FONT_HERSHEY_SIMPLEX,  
                              0.6, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(state_image, full_text1, (230,250), cv2.FONT_HERSHEY_SIMPLEX,  
                              0.6, (255,255,255), 1, cv2.LINE_AA) 
            cv2.imshow("Camera", cv2.vconcat([cv2.resize(state,(480,320)), state_image]))
            k = cv2.waitKey(1)
            if k==ord('p'):
                cv2.waitKey(0)
            
        # data l'azione sopra ottenuta, si ottiene la nuova immagine
        next_state, _, done = env.step(a)
  
        state = next_state
        step += 1

        # si interrompe quando si raggiunge il numero massimo di step
        if done or step > max_timesteps: 
            break
        

    

if __name__ == "__main__":                
    
    
    parser = argparse.ArgumentParser(description=
                                     "TESTING\n"
                                     "Default: --cones = False\n"
                                     "Aggiungere il tag --cones se si vuole eseguire il testing con i coni altrimenti verrà caricato il modello allenato sul tracciato strada.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('--cones', action='store_true', help='tipologia di tracciato')
    args = parser.parse_args()
    
    
    # numero di episodi 
    n_test_episodes = 15                  
    
    # istanza del modello
    agent = Model()
    
    
    #* Export the model
    # x = torch.randn(1, 1, 84,96)
    # torch.onnx.export(agent,               # model being run
    #                 x,                         # model input (or a tuple for multiple inputs)
    #                 "model.onnx",   # where to save the model (can be a file or file-like object)
    #                 export_params=True)


    #! Caricamento del modello ottimo ottenuto
    cones = args.cones # se cones = False viene caricato il modello road
    if cones:
        env = ConesEnv()
        agent.load("dagger_models/modelli ottimi/cones/multi_track_49_iter.pth",device)
    else:
        env = RoadEnv()
        agent.load("dagger_models/modelli ottimi/road/vel10_variabile.pth",device)
    
    # spostamento modello sul device selezionato
    agent.to(device)
  
    for i in range(n_test_episodes):
        run_episode()
        
     
    env.close()
    print('... finished')



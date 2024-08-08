import numpy as np
import draw_steering_angle
import torch
import cv2
from model import Model

#? Import ambienti
from environment_cones import ConesEnv
from environment_road import RoadEnv


#* Scelta del device di utilizzo: se è presente una GPU viene utilizzata
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print("Torch Device:", device)



# funzione usata per mostrare volante e velocità
#def draw_steer_speed(state):
    # color_rgb = env.getCamera_image()
    # bird_eye = cv2.resize(color_rgb, (480, 320))
    # punti per volante e barra verticale
    # pts = (np.array([[0, 0], [480, 0],
    #         [480, 70], [0, 70]],
    #         np.int32)).reshape((-1, 1, 2))
    # cv2.fillPoly(bird_eye, pts=[pts], color=(0, 0, 0))

    # steering_wheel = draw_steering_angle.SteeringWheel(bird_eye)
    
    # # disegno il volante per lo sterzo 
    # steering_wheel.draw_steering_wheel_on_image(a[0]*180/math.pi,(20,10))
    # # aggiungo la barra verticale per la velocità
    # vel_image = steering_wheel.update_frame_with_bar(a[1])
    
    # text = " rad/s"
    # full_text = f"{str(round(a[0],3))}{text}" 
    # text1 = " m/s"
    # full_text1 = f"{str(round(a[1],2))}{text1}" 
    
    # # Display del testo a video completi
    # cv2.putText(vel_image, full_text, (90,40), cv2.FONT_HERSHEY_SIMPLEX,  
    #                  0.6, (255,255,255), 1, cv2.LINE_AA) 
    # cv2.putText(vel_image, full_text1, (360,40), cv2.FONT_HERSHEY_SIMPLEX,  
    #                  0.6, (255,255,255), 1, cv2.LINE_AA) 
    
    # image_obs = state
    # image_obs = cv2.resize(state, (480, 320))
    # image_obs = cv2.cvtColor(image_obs, cv2.COLOR_GRAY2RGB)

    # cv2.imshow("Camera2", cv2.vconcat([vel_image, image_obs]))
    # cv2.waitKey(1) 



def run_episode(max_timesteps=2000):
    
    step = 0
    env.reset() # inizializzazione l'ambiente
    state= env.get_observation() # ottenimento delle osservazioni
    
    
    while True:
        # inizializzo un'immagine vuota per stampare in seguito i valori di velocità e sterzo
        state_image = np.zeros((320,480), dtype=np.uint8)
        
        # np.newaxis aumenta la dimensione dell'array di 1 (es. se è un array 1D diventa 2D)
        # torch.from_numpy crea un tensore a partire da un'array numpy
        # il modello ritorna le azioni (left/right, up/down)
        prediction = agent(torch.from_numpy(state[np.newaxis,np.newaxis,...]).type(torch.FloatTensor).to(device))
        a = prediction.detach().cpu().numpy().flatten()
        
        
        #! Scommentare le seguenti righe per mostrare il volante che sterza e la barra di velocità
        # draw_steer_speed(state)
        
        
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

    
        # data l'azione sopra ottenuta, si ottiene la nuova immagine
        next_state, _, done = env.step(a)
        #cv2.imshow("Camera", cv2.vconcat([cv2.resize(next_state,(480,320)), state_image]))
        cv2.imshow("Camera", cv2.resize(next_state,(640,480)))
        cv2.imshow("Camera2", cv2.cvtColor(env.getCamera_image(), cv2.COLOR_BGR2RGB))

        k = cv2.waitKey(1)
        if k==ord('p'):
            cv2.waitKey(0)
        
       
        state = next_state
        step += 1

        # si interrompe quando si raggiunge il numero massimo di step
        if done or step > max_timesteps: 
            break
        

    

if __name__ == "__main__":                
    
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
    cones = True # se cones = False viene caricato il modello road
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



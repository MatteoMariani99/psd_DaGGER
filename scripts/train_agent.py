import argparse
import optuna
import pickle
import numpy as np
import os
import gzip
import torch
from tqdm import tqdm
from model import Model
from torch.utils.data import DataLoader



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def read_data(datasets_dir="./data_test", path='data_dagger.pkl.gzip', frac = 0.1):
    """
    Funzione che permette di leggere i dati raccolti durante la simulazione e dividerli in 
    training e validazione.
    
    Parameters:
        - dataset_dir: directory contenente le immagini e le label ottenute
        - path: file contenente i dati da leggere
        - frac: frazione di split tra validazione e training
        
    Returns:
        - X_train: immagini per il training
        - y_train: label delle immagini per il training
        - X_valid: immagini per la validazione
        - y_valid: label delle immagini per la validazione
    """
    
    print("... read data")
    data_file = os.path.join(datasets_dir, path)
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # lettura imamgini e azioni (label)
    X = np.array(data["state"])
    y = np.array(data["action"]).astype('float32')

    # suddivisione dei dati
    # il 90% dei dati viene usato per il training e il 10% per la validation
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    
    return X_train, y_train, X_valid, y_valid



def train_model(X_train, y_train, path, num_epochs=20, learning_rate=1e-3, batch_size=32):
    """
    Funzione che permette di eseguire il loop per il training e salvare i pesi del modello allenato.
    
    Parameters:
        - X_train: immagini per il training
        - y_train: label per il training
        - path: path a cui si vuole salvare il modello allenato
        - num_epochs: numero di epoche per il training
        - learning rate: velocità di apprendimento 
        - batch_size: numero campioni di immagini del training per un forward pass della rete 
        
    Returns:
        - mean_loss: media della loss
    """
    print("... train model")
    model = Model()
    model.to(device)
    
    # definizione del dataloader utile poi per estrarre i campioni tramite batch size
    loader = DataLoader(dataset=list(zip(X_train, y_train)),batch_size=batch_size,shuffle=True)

    # criterio usato per il calolo della loss
    criterion = torch.nn.MSELoss()

    # ottimizzatore della rete per il calcolo del gradiente
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5) # built-in L2 

    loss_vector = []
    
    for t in tqdm(range(num_epochs)):
      for X_batch, y_batch in loader:
        preds  = model(X_batch[:,np.newaxis,...].type(torch.FloatTensor).to(device)) # prredizioni della rete
        loss   = criterion(preds, y_batch.to(device)) # calcolo della loss tra predette e vere
        
        # passo del gradiente
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_vector.append(loss)
    
    
    print("Loss training mean: ",(sum(loss_vector)/len(loss_vector)).detach().cpu().numpy().flatten())
    model.save(path)
    return (sum(loss_vector)/len(loss_vector)).detach().cpu().numpy().flatten()
    
    

def validate_model(X_valid, y_valid, model):
    """
    Funzione che permette di eseguire il loop per la validazione del modello.
    
    Parameters:
        - X_valid: immagini per la validazione
        - y_valid: label per la validazione
        - model: modello da validare
        
        
    Returns:
        - mean_loss: media della loss di validazione
    """
    # definizione del dataloader utile poi per estrarre i campioni tramite batch size
    loader = DataLoader(dataset=list(zip(X_valid, y_valid)),batch_size=1,shuffle=True)

    # criterio usato per il calolo della loss
    criterion = torch.nn.MSELoss()

    loss_vector = []
    print("... validate model")

    # utilizzo di torch.no_grad in quanto non devono essere aggiornati i pesi della rete
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds  = model(X_batch[:,np.newaxis,...].type(torch.FloatTensor).to(device))
            loss = criterion(preds, y_batch.to(device))
            loss_vector.append(loss)
            
   
    print("Loss validation mean: ",(sum(loss_vector)/len(loss_vector)).detach().cpu().numpy().flatten())
    return (sum(loss_vector)/len(loss_vector)).detach().cpu().numpy().flatten()
    
    
    
def objective(trial):
    """
    Funzione che permette di eseguire l'ottimizzazione dei parametri attraverso il framework OPTUNA
    
    Parameters:
        - trail: oggetto che contiene i parametri da ottimizzare
        
        
    Returns:
        - loss: calcolo della loss 
    """
    # parametri da ottimizzare:
    l_r = trial.suggest_float('learning_rate',1e-4,1e-1, log=True) # log= True in modo che varia il valore logaritmicamente
    batch_size = trial.suggest_categorical('batch_size',[16,32,64])
    num_epochs = trial.suggest_int('num_epochs',5,20)
    
    model = Model()
    model.to(device)

    # calcolo della loss tramite il loop di training
    loss = train_model(X_train, y_train, 'dagger_models/model_try_optim.pth', 
                       num_epochs=num_epochs, learning_rate=l_r,batch_size=batch_size)
    return loss


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description=
                                    "TRAINING\n"
                                    "Default: --optimize = False\t ottimizzazione parametri ",
                                    formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--optimize', action='store_true',help='processo di ottimizzazione')
    args = parser.parse_args()

    model = Model()
    model.to(device)
    
    # flag usato per la fase di ottimizzazione:
    # False: si esegue il training senza ottimizzazione parametri (senza OPTUNA)
    # True: si esegue il training con l'ottimizzazione dei parametri 
    optimize = args.optimize
 
    # lettura dei dati  
    X_train, y_train, X_valid, y_valid = read_data("./data_test", frac=0.1)
    
    
    if not optimize:
        loss = train_model(X_train, y_train, 'dagger_models/modelli ottimi/road/new_variabile_10.pth', num_epochs=20, learning_rate= 0.002044020588944688, batch_size=32)
        loss_val = validate_model( X_valid, y_valid, model)
    
    else:
        study = optuna.create_study(storage="sqlite:///db.sqlite3",direction='minimize')
        study.optimize(objective,n_trials=50)
        print(f'Best trial: {study.best_trial}')
        print(f'Best: {study.best_params}')
    
    
    #? è possibile visualizzare in real-time l'ottimizzazione dei parametri attraverso la dashboard optuna
    # optuna-dashboard sqlite:///db.sqlite3
    
# Best
#learning_rate': 0.002044020588944688, 
# 'batch_size': 32, 
# 'num_epochs': 20
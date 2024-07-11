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
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, path)
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"])
    y = np.array(data["action"]).astype('float32')


    # split data into training and validation set
    n_samples = len(data["state"])
    
    
    # il 90% dei dati viene usato per il training e il 10% per la validation
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid



def train_model(X_train, y_train, path, num_epochs=20, learning_rate=1e-3, batch_size=32):
    
    print("... train model")
    model = Model()
    model.to(device)
      
    loader = DataLoader(dataset=list(zip(X_train, y_train)),batch_size=batch_size,shuffle=True)

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5) # built-in L2 

    loss_vector = []
    for t in tqdm(range(num_epochs)):
      for X_batch, y_batch in loader:
        preds  = model(X_batch[:,np.newaxis,...].type(torch.FloatTensor).to(device))

        loss   = criterion(preds, y_batch.to(device))
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_vector.append(loss)
    
    
    print("Loss training mean: ",(sum(loss_vector)/len(loss_vector)).detach().cpu().numpy().flatten())
    model.save(path)
    return (sum(loss_vector)/len(loss_vector)).detach().cpu().numpy().flatten()
    
    
# valutare se calcolare la loss della validation: cerca costruzione early stop manuale 
def validate_model(X_valid, y_valid, model):
    
    
    loader = DataLoader(dataset=list(zip(X_valid, y_valid)),batch_size=1,shuffle=True)

    criterion = torch.nn.MSELoss()

    loss_vector = []
    print("... validate model")

    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds  = model(X_batch[:,np.newaxis,...].type(torch.FloatTensor).to(device))

            loss = criterion(preds, y_batch.to(device))
            loss_vector.append(loss)
            
   
    print("Loss validation mean: ",(sum(loss_vector)/len(loss_vector)).detach().cpu().numpy().flatten())
    return (sum(loss_vector)/len(loss_vector)).detach().cpu().numpy().flatten()
    
    
def objective(trial):
    l_r = trial.suggest_float('learning_rate',1e-4,1e-1, log=True) # log= True in modo che varia il valore logaritmicamente
    batch_size = trial.suggest_categorical('batch_size',[16,32,64])
    num_epochs = trial.suggest_int('num_epochs',5,20)
    model = Model()
    model.to(device)

    loss = train_model(X_train, y_train, 'dagger_test_models/model_try_optim.pth', 
                       num_epochs=num_epochs, learning_rate=l_r,batch_size=batch_size)
    return loss


if __name__ == "__main__":

    model = Model()
    model.to(device)
    
    optimize = False
 
    X_train, y_train, X_valid, y_valid = read_data("./data_test", frac=0.1)
    
    
    if not optimize:
        # utilizzo di params ottimi
        loss = train_model(X_train, y_train, 'dagger_test_models/modelli ottimi/cones/multi_track.pth', num_epochs=20, learning_rate= 0.002217342432015554, batch_size=16)
        loss_val = validate_model( X_valid, y_valid, model)
    
    else:
        #? for optimization hyperparams
        study = optuna.create_study(storage="sqlite:///db.sqlite3",direction='minimize')
        study.optimize(objective,n_trials=50)
        print(f'Best: {study.best_params}')
    
    
    #optuna-dashboard sqlite:///db.sqlite3
    
    # migliore:
    #Best: {'learning_rate': 0.002217342432015554, 
    # 'batch_size': 16, 
    # 'num_epochs': 20}


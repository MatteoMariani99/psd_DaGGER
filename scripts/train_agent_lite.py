import optuna
import pickle
import numpy as np
import os
import gzip
from model import Model
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from optuna.integration import PyTorchLightningPruningCallback


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
    
    

# classe utile per lutilizzo del modello Pytorch Lightning
class LitAgentTrain(pl.LightningModule):
    def __init__(self,learning_rate, batch_size,dataset_train, dataset_val):
        super().__init__()
        self.model = Model()
        self.learning_rate = learning_rate 
        self.train_dataset = dataset_train
        self.val_dataset = dataset_val
        self.batch_size = batch_size
        
    def training_step(self,batch,batch_idx):
        x,y = batch
        preds = self.model(x[:,np.newaxis,...].type(torch.FloatTensor).to(device))
        criterion = torch.nn.MSELoss()
        loss   = criterion(preds, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x[:,np.newaxis,...].type(torch.FloatTensor).to(device))
        criterion = torch.nn.MSELoss()
        val_loss   = criterion(preds, y)
        self.log("val_loss", val_loss, prog_bar=True)
        
    def predict_step(self, batch):
        x, y = batch
        return self.model(x)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=31)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=31)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)



def objective(trial):
    """
    Funzione che permette di eseguire l'ottimizzazione dei parametri attraverso il framework OPTUNA
    
    Parameters:
        - trail: oggetto che contiene i parametri da ottimizzare
        
        
    Returns:
        - loss: calcolo della loss 
    """
    # parametri da ottimizzare:
    l_r = trial.suggest_float('learning_rate',1e-4,1e-1,log=True)
    batch_size = trial.suggest_categorical('batch_size',[16,32,64])
    num_epochs = trial.suggest_int('num_epochs',5,20)
    
    # definizione del modello Pytorch Lightning
    light = LitAgentTrain(learning_rate=l_r, batch_size=batch_size, dataset_train=list(zip(X_train, y_train)), dataset_val=list(zip(X_valid, y_valid)))

    # definizione del loop (automatico) per il training
    trainer = pl.Trainer(max_epochs=num_epochs, 
                      accelerator='gpu', 
                      enable_checkpointing='False',
                      callbacks=[PyTorchLightningPruningCallback(trial,monitor="train_loss")])
    
    hyperparameters = dict(l_r = l_r,batch_size = batch_size,num_epochs = num_epochs)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(light)
    
    return trainer.callback_metrics["train_loss"].item()



if __name__ == "__main__":

    # read data
    X_train, y_train, X_valid, y_valid = read_data("./data_test", frac=0.1)
    
    pruner = optuna.pruners.HyperbandPruner()
    
    study = optuna.create_study(direction='minimize', pruner=pruner)
    study.optimize(objective,n_trials=20)
    print(f'Best trial: {study.best_trial}')
    print(f'Best: {study.best_params}')
    
    

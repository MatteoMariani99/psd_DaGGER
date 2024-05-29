import optuna
import pickle
import numpy as np
import os
import gzip
from tqdm import tqdm

from model import Model
from utils import *
import torch
import lightning.pytorch as pl
#from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
#from lightning.pytorch.tuner import Tuner
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sklearn.metrics import accuracy_score
from itertools import chain
from optuna.integration import PyTorchLightningPruningCallback


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



class LitAgentTrain(pl.LightningModule):
    def __init__(self,learning_rate, batch_size,dataset_train, dataset_val):
        super().__init__()
        self.model = Model()
        self.learning_rate = learning_rate 
        self.train_dataset = dataset_train
        self.val_dataset = dataset_val
        self.batch_size = batch_size
        #self.accuracy = accuracy_score(np.zeros(2),np.zeros(2))
        
    def training_step(self,batch,batch_idx):
        x,y = batch
        preds = self.model(x[:,np.newaxis,...].type(torch.FloatTensor).to(device))
        criterion = torch.nn.MSELoss()
        loss   = criterion(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x[:,np.newaxis,...].type(torch.FloatTensor).to(device))
        criterion = torch.nn.MSELoss()
        val_loss   = criterion(preds, y)
        #print(type(preds))
        #print(type(y))
        #print("ok!", list(chain(*preds.detach().cpu().numpy().astype(int))))
        #print("o",list(chain(*y.detach().cpu().numpy().astype(int))))
        self.log("val_loss", val_loss, prog_bar=True)
        #self.log("accuracy", self.accuracy(list(chain(*preds.detach().cpu().numpy().astype(int))),list(chain(*y.detach().cpu().numpy().astype(int)))), prog_bar=True)
        
    def predict_step(self, batch):
        x, y = batch
        return self.model(x)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=31)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=31)

    
    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)



def train_model(X_train, y_train, path, optimizer, num_epochs=50, learning_rate=1e-3, batch_size=32):
    
    print("... train model")
    model = Model()
    model.to(device)
      
    loader = DataLoader(dataset=list(zip(X_train, y_train)),batch_size=batch_size,shuffle=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) # built-in L2 

    loss_vector = []
    for t in tqdm(range(num_epochs)):
      for X_batch, y_batch in loader:
        preds  = model(X_batch[:,np.newaxis,...].type(torch.FloatTensor).to(device))
        
        loss   = criterion(preds, y_batch.to(device))
        #print("Loss: ",loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_vector.append(loss)
    
    print("Loss mean: ",(sum(loss_vector)/len(loss_vector)).detach().cpu().numpy().flatten())
    model.save(path)
    return (sum(loss_vector)/len(loss_vector)).detach().cpu().numpy().flatten()
    
    
class ModelProva(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super(ModelProva, self).__init__()
        self.model = Model()  # Replace with your actual model class
        self.criterion = torch.nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        preds = self.model(X_batch[:, np.newaxis, ...].type(torch.FloatTensor).to(self.device))
        loss = self.criterion(preds, y_batch.to(self.device))
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        return optimizer


def objective(trial):
    l_r = trial.suggest_float('learning_rate',1e-4,1e-1,log=True)
    batch_size = trial.suggest_categorical('batch_size',[16,32,64])
    num_epochs = trial.suggest_int('num_epochs',5,20)
    #model = Model()
    #model.to(device)
    #X_train, y_train, X_valid, y_valid = read_data("./data_test", frac=0.1)
    
    light = LitAgentTrain(learning_rate=l_r, batch_size=batch_size, dataset_train=list(zip(X_train, y_train)), dataset_val=list(zip(X_valid, y_valid)))

    logger = TensorBoardLogger("tb_logs", name="my_model")
    trainer = pl.Trainer(max_epochs=num_epochs, 
                      accelerator='gpu', 
                      logger=logger, 
                      enable_checkpointing='False',
                      callbacks=[PyTorchLightningPruningCallback(trial,monitor="train_loss")])
    
    hyperparameters = dict(l_r = l_r,batch_size = batch_size,num_epochs = num_epochs)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(light)
    
    #loss = train_model(X_train, y_train, X_valid, y_valid, 'dagger_test_models/model_0.pth', num_epochs=num_epochs, learning_rate=l_r,batch_size=batch_size)
    return trainer.callback_metrics["train_loss"].item()



if __name__ == "__main__":

    # read data
    X_train, y_train, X_valid, y_valid = read_data("./data_test", frac=0.1)
    
    pruner = optuna.pruners.HyperbandPruner()
    
    study = optuna.create_study(direction='minimize', pruner=pruner)
    study.optimize(objective,n_trials=20)
    print(f'Best trial: {study.best_trial}')
    print(f'Best: {study.best_params}')
    
    #loader_train = DataLoader(dataset=list(zip(X_train, y_train)),shuffle=True)
    #loader_val = DataLoader(dataset=list(zip(X_valid, y_valid)),shuffle=True)

    # light = LitAgentTrain(learning_rate=1e-3, batch_size=32, dataset_train=list(zip(X_train, y_train)), dataset_val=list(zip(X_valid, y_valid)))

    # logger = TensorBoardLogger("tb_logs", name="my_model")
    # trainer = Trainer(max_epochs=10, 
    #                   accelerator='gpu', 
    #                   logger=logger, 
    #                   callbacks=EarlyStopping(monitor="val_loss"),
    #                   enable_progress_bar=True)
    
    # trainer.fit(model=light)
    # tuner = Tuner(trainer)
    #tuner.scale_batch_size(light,mode='power',init_val=16)
    #tuner.lr_find(light,min_lr=1e-5, max_lr=1e-2)
    # lr_finder = trainer.tuner.lr_find(light)
    # new_lr = lr_finder.suggestion()
    # light.learning_rate = new_lr
    
    # # Find optimal batch size
    # new_batch_size = trainer.tuner.scale_batch_size(light)
    # light.batch_size = new_batch_size

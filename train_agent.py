from __future__ import print_function
import argparse
import pickle
import numpy as np
import os
import gzip
from tqdm import tqdm

from model_new import VehicleControlModel
from utils import *
import torch
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vel_max = 15

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


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can either train your model with continous actions (as you get them from read_data) using regression
    #    or you discretize the action space using action_to_id() from utils.py. If you discretize them, you'll maybe find one_hot() 
    #    useful and you may want to return X_train_unhot ... as well.
    #print(X_train.shape)
    
    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96,1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).
    #X_train = rgb2yuv(X_train)
    #X_valid = rgb2yuv(X_valid)
    array_train = []
    array_valid = []
    #print(X_valid.shape)

    for i in range(len(X_train)):
        gray_train = cv2.cvtColor(X_train[i,:,:,:],cv2.COLOR_RGB2YUV)
        cv2.imshow('Original', gray_train) 
        cv2.waitKey(1)
        array_train.append(gray_train)   
        
    for j in range(len(X_valid)):
        gray_valid = cv2.cvtColor(X_valid[j,:,:,:],cv2.COLOR_RGB2YUV)
        array_valid.append(gray_valid)
        
    #X_train = (X_train)[:,:CUTOFF,:]
    #X_valid = (X_valid)[:,:CUTOFF,:]
    #samples["state"].append(state)  
    X_train = np.array(array_train)
    X_valid = np.array(array_valid)
    
    #print(X_train.shape)
    #print(X_valid.shape)
    
    return X_train, y_train, X_valid, y_valid


def train_model(X_train, y_train, X_valid, y_valid, path, num_epochs=50, learning_rate=3e-4, lambda_l2=1e-5, batch_size=32):
    
    print("... train model")
    model = VehicleControlModel()
    model.to(device)
    #print(model.get_device())
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lambda_l2) # built-in L2 
    #cv2.imshow('Original', X_train[100,:,:,:]) 
    #cv2.waitKey(0)
    X_train_torch = torch.from_numpy(X_train).to(device)
    y_train_torch = torch.from_numpy(y_train).to(device)
    
    
    #print(X_train_torch.shape)
    
    X_train_torch = (X_train_torch.permute(0,3,1,2))
    
    #print(X_train_torch.shape)
    

    for t in tqdm(range(num_epochs)):
      #print("[EPOCH]: %i" % (t), end='\r')
      for i in range(0,len(X_train_torch),batch_size):
        curr_X = X_train_torch[i:i+batch_size]
        
        curr_Y = y_train_torch[i:i+batch_size]

        preds  = model(curr_X.type(torch.FloatTensor).to(device))
        #print(f"Action prede: {preds} Action curr: {curr_Y}")
        
        loss   = criterion(preds, curr_Y)
        #print("Loss: ",loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    model.save(path)


# manca la validazione -> basta richiamare il modello e verificare le predizioni ottenute (verifico le predizioni in uscia
# e le confronto con le y_valid)
if __name__ == "__main__":

    #parser = argparse.ArgumentParser()
    #parser.add_argument('model_name', metavar='M', default='model.pth', type=str, help='model name to save')
    #args = parser.parse_args() 
    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data_test", frac=0.1)
    
    
    # preprocess data
    #X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=1)
    # train model
    train_model(X_train, y_train, X_valid, y_valid, 'dagger_test_models/model_5.pth', num_epochs=10)
 
  

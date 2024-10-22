import torch
import torch.nn as nn

# Implementazione del modello tramite il framework PyTorch
class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1,4,5)
        self.norm1 = nn.BatchNorm2d(4) # normalizzo in modo da non aver divergenza dei parametri
        self.conv2 = nn.Conv2d(4,8,5)
        self.norm2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8,16,6)
        self.norm3 = nn.BatchNorm2d(16)
        self.lin1  = nn.Linear(768,64)  # sono i dense layer
        self.lin2  = nn.Linear(64,16)
        self.lin3  = nn.Linear(16,2)

        self.maxpool = nn.MaxPool2d(2) # riduce la dimensionalità della rete (riduce il rischio di overfitting)
        self.dropout = nn.Dropout2d(0.1) # previene l'overfitting. 0.1 è la probabilità con cui eliminare i canali
        self.relu    = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self,x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.maxpool(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.maxpool(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.maxpool(x)
        x = self.relu(x)


        x = self.flatten(x)


        x = self.lin1(x)
        x = self.relu(x)
       
        x = self.lin2(x)
        x = self.relu(x)
        
        return self.lin3(x)
    
    def load(self, path, device):
        self.load_state_dict(torch.load(path,map_location=device))

    def save(self, path):
        torch.save(self.state_dict(), path)



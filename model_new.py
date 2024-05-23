import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import cv2

class VehicleControlModel(nn.Module):
    def __init__(self):
        super(VehicleControlModel, self).__init__()

        # per trovare la dimensione in uscita ai layer fare
        #((size - kernel_size) / (stride)) + 1
        # se non si trova un intero, si arrotonda
        
        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.flatten = nn.Flatten()
        
        # Define fully connected layers
        # l'uscita dal layer convoluzionale presenta una dimensione pari a 1x18 (dimensione immagine) x l'output del layer (64)
        # 1164 sono quelli che voglio io
        self.fc1 = nn.Linear(64 * 1 * 18, 1164)
        self.fc2 = nn.Linear(1164, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        self.fc5 = nn.Linear(10, 2) # proviamo solo con sterzo e forward (positiva avanti, negativa indietro)
        
        # Normalization layer
        # batch norm2d in quanto entra un tensore quindi di dimensione 4
        #self.norm = nn.BatchNorm2d(3)
        self.norm = nn.BatchNorm2d(1)

    def forward(self, x):

        # Apply normalization
        x = self.norm(x)
        
        # Apply convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        # Flatten the output from the convolutional layers
        x = self.flatten(x)
        
        # Apply fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        
        # Output layer
        x = self.fc5(x)
        # tanh per sterzo (-1,+1)
        # 
        #x = torch.reshape(torch.cat([torch.tanh(x[:, 0]), self.vel_max*torch.tanh(x[:, 1])], dim=0),x.shape)
        
        # Output layer with Tanh activation: tnah per sterzo (-1,+1 destra,sinistra) e per velocità (-1 indietro +1 avanti)
        # moltiplicare il tanh per velocità che si vuole avere con la macchina, per lo sterzo va bene 1;-1
        #x = torch.tanh(self.fc5(x))
        
        return x

    def load(self, path):
        self.load_state_dict(torch.load(path,map_location="cuda:0"))

    def save(self, path):
        torch.save(self.state_dict(), path)

# Instantiate the model
#model = VehicleControlModel()

# Print the model architecture
#print(summary(model,(3,66,200)))

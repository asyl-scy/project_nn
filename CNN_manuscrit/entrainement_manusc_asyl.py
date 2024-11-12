"""
@author = asyl_scy (a.k.a Kara)

date = 12/11/2024

topic = "est-ce qu'asyl_scy a une écriture si horrible que ça?"

ici, je vais tester mon horrible (?) ecriture: on va savir si le réseau de neurone que j'ai entrainé
reconnait mon écriture.
"""


import torch
#--------------------réseau de neurones
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim#pr initialisation fonction de perte et optimizer
#----------------------image
import torchvision
from torchvision import datasets, transforms
from PIL import Image

#----------------------les bros
import matplotlib.pyplot as plt
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super().__init__()#encore lui!! sert à avoir les fonctionnalités de nn.Module
        self.conv1 = nn.Conv2d(1, 6, 5)#il n'y a qu'un canal de couleur pour le dataset emnist, pas 3
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        #calcul
        self._calculate_fc1_input_dim()

        self.fc1 = nn.Linear(self.fc1_input_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 26)

    def _calculate_fc1_input_dim(self):
        #fonction qui va servir à calculer la taille que les images ont après application des convolutions
        with torch.no_grad():
            # Crée un tenseur fictif de la même taille que les images d'entrée (1, 28, 28)
            x = torch.zeros(1, 1, 28, 28)  # Taille d'entrée pour une image issu de EMNIST
            x = self.pool(F.relu(self.conv1(x)))#pool n°1
            x = self.pool(F.relu(self.conv2(x)))#pool n°2
            self.fc1_input_dim = x.numel()  # Nombre total d'éléments dans le tenseur aplati

    def forward(self, x):
        # fonction qui comporte le réseau de neurone (les pooling/convolutions, la couche flatten et les couches à activation relu)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def transform_torch(image,transform):
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

#on va commencer par importer les images ainsi que le réseau de neurone entrainé plus tôt
#------------image
path= "data/ecriture"
image_a= Image.open(path+"/ecriture_asyl_a.png").convert("L")#je veux que ça soit en echelle de gris, donc j'ajoute convert("L")
image_f=Image.open(path+"/ecriture_asyl_f.png").convert("L")
image_l=Image.open(path+"/ecriture_asyl_l.png").convert("L")

#tranformation image en objet pytorch

transform = transforms.Compose([
    transforms.ToTensor(),  # Convertit l'image en un tenseur PyTorch//sert à convertir mes image PIL en tenseur pytorch
    transforms.Normalize((0.5,), (0.5,))  # normalisation
])

image_a_tensor=transform_torch(image_a,transform)
image_f_tensor=transform_torch(image_f,transform)
image_l_tensor=transform_torch(image_l,transform)



#-------------nn
net=Net()
net.load_state_dict(torch.load("./training_manu_asyl.pth", weights_only=True))#pour load des models



#là, on va voir si le réseau de neurone reconnait mon écriture

print("------------prediction a")

outputs = net(image_a_tensor)

_, predicted = torch.max(outputs, 1)
print(predicted)

print("------------prediction l")

outputs = net(image_l_tensor)

_, predicted = torch.max(outputs, 1)
print(predicted)#d'après mon nn, c'est un x!

print("------------prediction f")

outputs = net(image_f_tensor)

_, predicted = torch.max(outputs, 1)
print(predicted)#genre ça c'est un a!?

#....ahaha, non mais c'ets sûrement par ce que j'ai choisi comme par hasard 2 lettres que mon nn a du mal à deviner...non?!!!
#(réponse dans la partie 2)
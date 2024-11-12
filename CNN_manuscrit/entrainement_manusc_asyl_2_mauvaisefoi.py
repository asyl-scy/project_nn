"""
@author = asyl_scy (a.k.a Kara)

date = 07/11/2024

topic = "est-ce qu'asyl_scy a une écriture si horrible que ça?"

...non mais en vrai, c'est sûrement un problème de prediction, non?

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

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def adjust_labels(target):#fonction qui va servir à ne pas contrarier l'entropie (quand il y a plus de 25 labels, marche plus)
    return target - 1

#chargement image EMNIST
transform = transforms.Compose([
    transforms.ToTensor(),  # Convertit l'image en un tenseur PyTorch//sert à convertir mes image PIL en tenseur pytorch
    transforms.Normalize((0.5,), (0.5,))  # normalisation
])

# Télécharger et charger le dataset EMNIST (sous-ensemble "letters" pour les lettres manuscrites)
dataset = datasets.EMNIST(
    root="data",
    split="letters",  # Choix du split "letters" pour les lettres manuscrites
    train=True,  # Chargement de l'ensemble d'entraînement
    download=True,
    transform=transform,
    target_transform = adjust_labels  # Applique adjust_labels aux labels
)

#-------------nn
net=Net()
net.load_state_dict(torch.load("./training_manu_asyl.pth", weights_only=True))#pour load des models

#on va couper la poire en 2 pour faire notre test

data_use_int = int(0.5 * len(dataset))
data_use,_ = torch.utils.data.random_split(dataset, [data_use_int, data_use_int])
data_use_loader = torch.utils.data.DataLoader(data_use, batch_size=64, shuffle=True)

#listes qui vont stocker les résultats (bonne prediction et le nombre de prédiction tot) pour chaque classes (A-Z)
class_correct = [0 for _ in range(26)]
class_total = [0 for _ in range(26)]

data_use_iter=iter(data_use_loader)
with torch.no_grad():
    for data in data_use_loader:
        images, labels = data#recup_re image et label
        outputs = net(images)#prédiction par le nn
        _, predicted = torch.max(outputs, 1)#recupération du label prédit par le nn
        for i in range(len(labels)):
            label = labels[i].item()
            if predicted[i] == label:
                class_correct[label] += 1
            class_total[label] += 1

# Affichage de la précision pour chaque classe
for i in range(26):
    if class_total[i] > 0:
        accuracy = 100 * class_correct[i] / class_total[i]
        print(f'Accuracy of class {chr(i + ord("A"))}: {accuracy:.2f}%')

#bon ,il est pas très compétent pour le l (78.10% de réussite) mais l'est bien plus pour le f (93.28%)

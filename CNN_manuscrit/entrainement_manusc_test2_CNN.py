"""
@author = asyl_scy (a.k.a Kara)

date = 07/11/2024

topic = "est-ce qu'asyl_scy a une écriture si horrible que ça?"

On va entrainer grâce au jeu de donnée EMNIST le réseau de neurone qui va juger mon horrible(?) écriture.
"""



#test_

import torch
#--------------------réseau de neurones
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim#pr initialisation fonction de perte et optimizer
#----------------------image
import torchvision
from torchvision import datasets, transforms
#----------------------les bros
import matplotlib.pyplot as plt
import numpy as np

def save_model(save):
    #fonction qui va servir à sauvegarder les poids etc du réseau de neurone.
    if save:
        PATH = './training_manu_asyl.pth'
        torch.save(net.state_dict(), PATH)

def imshow(img):
    #fonction pour afficher des images
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

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

# Transformation des labels pour qu'ils soient dans la plage [0, 25]
def adjust_labels(target):#fonction qui va servir à ne pas contrarier l'entropie (quand il y a plus de 25 labels, marche plus)
    return target - 1


def plot_loss_fct(num_epoch, loss_history):#va servir à plot la fonction de perte
    plt.plot(range(1, num_epoch + 1), loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Evolution de la fonction de perte au cours des epochs")
    plt.legend()
    plt.show()


# Définir les transformations pour les images
transform = transforms.Compose([
    transforms.ToTensor(),  # Convertit l'image en un tenseur PyTorch//sert à convertir mes image PIL en tenseur pytorch
    transforms.Normalize((0.5,), (0.5,))  # Normalisation des valeurs des pixels//"réduit" les images pour faciliter l'apprentissage
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


#définir taille pour test et training
train_size = int(0.8 * len(dataset))  # 80% pour l'entraînement
test_size = len(dataset) - train_size  # 20% pour le test

net=Net()

#séparation du dataset en training et test
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

#créer un dataloader pour itérer les 2 jeux de données
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

#on initialise optimizer ainsi que la fonction de perte

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#ce qui est (-ouf-)passionnant avec pytorch, c'est qu'on doit faire l'epoch nous-même!!
#num_epoch=1000
num_epoch=4
loss_history = []#ce qui va stocker les résults
save=True
for epoch in range(num_epoch):
    running_loss=0.0
    for i, data in enumerate(train_loader, 0):#i itère train_loader et data 0?
        #print(data)

        #on sépare les inputs et labels qui sont dans data
        inputs, labels = data

        # mets le gradient à zero
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)#fonction de perte
        loss.backward()
        optimizer.step()

        #print(f"epoch {epoch + 1} : loss :{} / optimizer = ")
        # print statistiques
        running_loss += loss.item()
    #calculs fonction de perte
    avg_loss = running_loss / len(train_loader)
    loss_history.append(avg_loss)

    # Affichage de l'epoch, de la perte moyenne et de l'état de l'optimiseur
    print(f'Epoch {epoch + 1}/{num_epoch}, Loss: {avg_loss:.4f}')
    for param_group in optimizer.param_groups:
        #lr= learning_rate/
        print(f"Optimizer State: lr = {param_group['lr']}, momentum = {param_group.get('momentum', 0)}")

print('Fin entrainement')

save_model(save)#sauvegarder model
plot_loss_fct(num_epoch, loss_history)#afficher fct perte

#----------------validation
#score
correct = 0# Compteur prédiction corrects
total = 0#compteur du nombre total d'images

dataiter_test = iter(test_loader)
images_test, labels_test = next(dataiter_test)

#net = Net()
#net.load_state_dict(torch.load(PATH, weights_only=True))#pour load des models

net.eval()#mettre net en mode évaluation

with torch.no_grad():#on enlève le cacul de gradient pour faciliter les calculs
    for data in test_loader:
        images, labels = data #recup image et label
        outputs = net(images) #obtenir les sorties du modèle

        # obtenir prediction
        _, predicted = torch.max(outputs, 1)#on prend le label avec la plus haute proba (e vrai, on aurait pu mettre une couche softmax pour faciliter cet étape)

        # Màj compteurs
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculer et afficher la précision
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy:.2f}%')




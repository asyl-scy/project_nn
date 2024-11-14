# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 04:53:36 2024

@author: asyl_scy (Kara)

topic = "est-ce que un réseau de neurone peut deviner les facteurs d'une fonction?"
"""

import tensorflow as tf
from tensorflow.keras import layers, models

import numpy as np
from scipy import stats
import scipy.stats as st 
import matplotlib.pyplot as plt
import random

from sklearn.preprocessing import StandardScaler


#méthode 2 adaptable
def recur_fct_vrai(choix,x,score=0,puissance=1,stock=0):
    """
    fonction qui va calculer une fonction mathématique. Fait pour se débrouiller pour n'importe quel degré'

    Parameters
    ----------
    choix : liste
        facteurs.
    x : int
        la valeur de x.
    score : int, optional
        compte à rebours. mais à initialiser! The default is 0.
    puissance : int, optional
        A ne pas toucher. The default is 1.
    stock : int, optional
        A ne pas toucher. The default is 0.

    Returns
    -------
    stock : int
        l'image de la fonction.

    """
    #mon bébé marche!!
    
    if score>=0:
        stock=choix[score]*x**(puissance)+stock
        score=score-1
        puissance+=1
        stock=recur_fct_vrai(choix,x,score,puissance,stock)
    return stock


def plot_scatter(stock):
    #plot les erreurs
    plt.figure()
    plt.scatter(np.arange(len(stock)),stock)
    plt.figure()
    plt.hist(stock)


def generate_data(nb_choix,x,num_samples=1000):
    """
    

    Parameters
    ----------
    nb_choix : int
         degrés de l'équation
    x : liste
        différents x testés
    num_samples : TYPE, optional
        nombre d'échantillons. The default is 1000.

    Returns
    -------
    listes (X et Y)
        ensemble des facteurs (X) et des images associés (Y) .

    """
    X = []
    y = []
    for _ in range(num_samples):
        choix = np.round(np.random.rand(nb_choix),2)  # coefficients aléatoires
        
        y_val=[recur_fct_vrai(choix,ola,score=nb_choix-1)for ola in x ]
        X.append(y_val)  # entrée pour le réseau
        y.append(choix)  # cible pour le réseau
    #return torch.tensor(X, dtype=torch.float32).view(-1, 1), torch.tensor(y, dtype=torch.float32)
    return np.round(np.array(X, dtype=np.float32),2), np.array(y, dtype=np.float32)

#----------données initiales
nb_choix=4#nb choix
x = [-10,-5,-1,2,3,6,10,20]

X_train,y_train=generate_data(nb_choix,x)

#----------------modele nn
#creation nn+ optimizer+ perte
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(len(x),)),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),#on en rajoute 1 car le RN n'arrive pas à prédire corectement
    layers.Dense(nb_choix)  # 4 pour choix + 1 pour x
])
model.compile(optimizer='adam', loss='mse')

#----------------entrainement
model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=1)
print("Entraînement terminé")


#----------------tentative de calcul de précision
"""j'ai envie d'avoir une information quantitative qui me permettra de juger la précision de mon nn.
Pour ce faire, je vais calculer une marge d'erreur.
Je ne peux pas le faire pour une seule fonction, donc j'en fait pour 1000.
Pour chacune d'entre elles, je calcule l'erreur moyenne entre les points de la véritable fonction et celle
que le réseau de neurone aura deviné. 
On fera des opérations sur ces erreurs moyennes pour obtenir la marge d'erreur. C'est la strat que j'ai trouvé
pour le moment.

"""
#calcul erreur moyenne
stock_mean=[]
print("calcul erreur moyenne")
n=1001
for _ in range(n):
    print(f"prediction numero {_} :")
    #initialisation facteurs
    choix = np.round(np.random.rand(nb_choix),2)
    #calcul des Y des fonctions selon plusieurs x
    inputs=np.round(np.array([recur_fct_vrai(choix,ola,score=nb_choix-1)for ola in x ]),2)
    #reshape pour qu'il soit exploitable par keras
    inputs = inputs.reshape(1, -1) 
    #prediction des facteurs à partir des outputs par le nn
    essai=model.predict(inputs)
    
    #construction de la courbe des fonctions
    x_simu=np.linspace(-5,5)
    simu_nn=[recur_fct_vrai(essai[0],x,score=nb_choix-1) for x in x_simu]
    simu_vrai=[recur_fct_vrai(choix,x,score=nb_choix-1) for x in x_simu]
    
    #on va créer une métrique qui va considérer l'erreur du ML
    error_sus=[simu_nn[a]-simu_vrai[a] for a in range (len(x_simu))]
    stock_mean.append(np.mean(np.abs(error_sus)))


print("fin multi_simulation")

#plot erreurs(scatter)
plot_scatter(stock_mean)

#on va tester la normalité pour savoir comment m'y prendre pour avoir les 95%CI
#shapiro test = un autre moyen de tester la normalité de mon jeu de donnée => si pval<0.05, normal (pour python)
res = stats.shapiro(stock_mean)
print(f"le résultat du test de shapiro est {res.statistic} \n")
if res.statistic < 0.05:
    print("notre jeu de donnée suis une distribution normale \n")
    normal=True
else:
    print("notre jeu de donnée ne suis pas une distribution normale \n")
    normal=False


if normal==False:
    
    #test boostrap
    bootstrap_means = []
    for _ in range(1000):
        sample = np.random.choice(stock_mean, size=len(stock_mean), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    
    # Calcul de l'intervalle de confiance à 95 %
    lower_bound = np.percentile(bootstrap_means, 2.5)
    upper_bound = np.percentile(bootstrap_means, 97.5)
    
    
    print(f"Intervalle de confiance à 95 % (bootstrap) : [{lower_bound:.2f}, {upper_bound:.2f}] \n")
    
    #n.b= il se peut que la technique du bootstrap ne fonctionne pas trop dans ce contexte, on vérifie ça
    #verification intervalle confiance
    seuil_quantite=5*len(stock_mean)/100#seuil à mieux définir
    quant_inf_value=len([x for x in stock_mean if x<lower_bound])/len(stock_mean)*100
    quant_sup_value=len([x for x in stock_mean if x>upper_bound])/len(stock_mean)*100
    print(f"avec une valeur de {quant_inf_value:.2f} % de données inférieurs à {lower_bound:.2f} on peut considérer que la méthode boostrap est compliquée \n" if quant_inf_value < seuil_quantite else f"avec une valeur de {quant_inf_value:.2f} % de données inférieurs à {lower_bound:.2f} on peut considérer que la méthode boostrap est une méthode fiable pour l'intervalle de confiance \n")#proportion d'élément plus petits que le lower_bound
    print(f"avec une valeur de {quant_sup_value:.2f} % de données supérieurs à {upper_bound:.2f}on peut considérer que la méthode boostrap est compliquée \n" if quant_sup_value < seuil_quantite else f"avec une valeur de {quant_sup_value:.2f} % de données supérieurs à {upper_bound:.2f} on peut considérer que la méthode boostrap est une méthode fiable pour l'intervalle de confiance \n")#proportion d'élément plus petits que le lower_bound


    if quant_inf_value < seuil_quantite or quant_sup_value < seuil_quantite :
        #si la méthode boostrap est pas pertinent, on fait avec la médiane
        lower_bound_med = np.percentile(stock_mean, 2.5)
        upper_bound_med = np.percentile(stock_mean, 97.5)
        print(f"Intervalle de confiance à 95 % (médiane) : [{lower_bound_med:.2f}, {upper_bound_med:.2f}] \n")

#95CI si normal
elif normal==True:
    print("la distribution est normal")#
    int_conv=st.norm.interval(alpha=0.95, 
                 loc=np.mean(stock_mean), 
                 scale=st.sem(stock_mean)) 
    print(int_conv)

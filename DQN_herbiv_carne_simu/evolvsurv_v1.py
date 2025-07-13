# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 19:42:15 2025

@author: asyl_scy (Kara)

topic= projet evolusurv

version 1.0 
"""



import pygame
from pygame.locals import*
import sys
import random
import numpy as np
from random import randint
import matplotlib.pyplot as plt


from torch import nn
import torch

import itertools
from collections import deque


def warn_training(len_value,name_animal):
    
    """
    Fonction (imparfaite)qui sert a plus ou moins savoir quand la phase "training" commence 
    
    len_value=nombre d'individu dans l'espèce
    name_animal= nom de l'espèce
    """
    
    if len_value==BEGIN_TRAIN:
        print(f"ça commence pour {name_animal}")

def show_grid():
    """sert à montrer la grille"""
    for i in range(0, NB_COL):
        for j in range(0, NB_ROW):
            rect = pygame.Rect(i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, pygame.Color("grey"), rect, width=1)


def id_stp1_transi_manage(id_st,animal,stp1):
    
    """
    repère le STP1 correspondant à un st, sinon renvoie un vecteur nul
    
    id_st= ID de l'état
    animal= la classe de l'espèce
    stp1= la state en temps +1
    
    """
    
    if id_st in animal.record_ID.tolist():
        ind=np.where(np.isin(animal.record_ID,id_st))[0]#recup indice où l'id de stp1 correspond à st
        #print(ind)
        return stp1[ind[0]] 
    else:
        return np.zeros(np.shape(stp1)[1])

def reward_done(ID,animal,where_repro,HP_past):
    
    """
    gère les récompenses
    
    ID= identifiant
    animal= classe de l'epèce
    where_repro= vecteur contenant les individus qui ont fait une reproduction
    HP_past= HP des individus au temps 0 (non au temps +1)
    
    
    """
    
    #animal= simul.chaine.herbiv ou simul.chaine.carne
    #liste des éléments ayant fait une reproduction
    #HP de l'individu
    
    #rew_record=np.zeros(len(rew))#comparaison ID
    rew=0
    #mourir

    #reproduction
    if len(where_repro)!=0:
        repro_ID=[ID for i in where_repro[0]]
        #print(len(repro_ID))
        #print(repro_ID)
        if len(repro_ID)!=0 and ID in repro_ID:#.tolist()
            #gratifie la reproduction
            rew+=2#c'était 3 de base
    
    if ID not in animal.record_ID:
        #punit la mort
        rew-=2
        return [rew, True]
    
    
    
    
    #pour avoir accès à l'HP actuel
    arg_element=np.where(np.any(animal.record_ID==ID))
    
    #gagne ernegie (donc bouffe non abusive)
    if animal.record_energy[arg_element]-HP_past<0:#résultats négatif= gain d'énergie
        rew+=1
    
    #punit la perte d'énergie
    elif animal.record_energy[arg_element]-HP_past>=10:#résultat positif= perte d'énergie
        #repro( perte contrebalancée) (+ se faire bouffer pour les herbiv (perte non contrebalancée))
        rew-=1
        
    # elif animal.record_energy[arg_element]-HP_past==-1:#puni les mouvements inutiles
    #     rew-=0.5
    
    
    
    return[rew,False]
    
    
    #se fait bouff(que pour les herbiv)
        

        
    pass
    
    
    

    
def max_energy(age_switch,input_age, list_energy, energy_t0 ):
    
    """
    age_switch: l'age où sa switch
    stock_age: les ages
    list_energy: dans la liste=[age_début, age_fin]
    """
    
    if input_age==list_energy:
        energy=energy_t0
    elif input_age<=age_switch:
        pass
        #energy=





def take_action(animal,nb_action,eps=1,DQN=None,st=None):#on n'utilisera pas celui-là

        """
        
        eps=epsilon-greedy
        DQN=le modèle qui pourrait être utilisé.
        st=l'état
        (prendra la dqn de la classe, on aurait pas besoin de mettre dqn en param)
        
        """
        
        ##print(dir(animal))
        
        for row in range(len(animal.record_action)):

            if random.uniform(0, 1) < eps:
                animal.record_action[row] = randint(0, animal.nb_action-1)#on veut pas inclure 12
            elif DQN!=None and st!=None : # Or greedy action
                #print("eps:",eps)
                animal.record_action[row]  = DQN.act(st[row])#selectionne action qui maximise l'espérance à l'état st 
     
                
def case_to_pixel(case):
    """
    va récupérer la couleur (RGB) d'une pixel.
    case= position de la pixel
    """
    
    return screen.get_at(case)[:3] 

def vision_to_pixel(vision):
    """
    va prendre un champ de vision et retourner une liste de couleurs
    
    vision= "champ de vision", liste de positions
    
    """
    list_f=[]
    for i in vision:
        for k in i:
            #print(k)
            if -10 in k:
                #valeur aberrente s'il y a une coordonnée hors du jeu
                list_f.append(-10)
                continue
            # #print(k)
            # #print(case_to_pixel(k))
            # #print("---"*6)
            
            list_f.append(gray_scale(case_to_pixel(k)))
    return list_f

def gray_scale(rgb):
    """
    transforme une matrice rgb en valeur nuance de gris
    rgb= liste (R,G,B)
    """
    r,g,b = rgb
    return int(0.299 * r + 0.587 * g + 0.114 * b)
                

def done_ask(id_t, ids_tp1):
    """
    permet de gérer les "done" des states
    id_t= id des states
    ids_tp1=id des states au temps +1
    """
    if id_t in ids_tp1:
        return False
    return True


class Network(nn.Module):
    
    """
    classe réseau de neurone (pytorch)
    """
    def __init__(self,info_transition):
        
        #info_transition[0]=len(st), info_transition[1]=len(action)
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(info_transition[0], 128),
            nn.Tanh(),#Faudrait aussi tenter avec un simple ReLU
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128,info_transition[1])
            )
        
    def forward(self,x):#si pas présent, bug
        return self.net(x)
    
    def act(self,st):

        """
        fonction qui va retournée la meilleure action
        st=l'état.
        """
        #print("Shape de st dans act:", np.shape(st))
        #récupère l'action avec la meilleur q_value
        st_t=torch.as_tensor(st,dtype=torch.float32)#torch tensor
        q_value=self(st_t.unsqueeze(0))#t'ajoute un plan
        #on prend l'action avec la meilleure q_value
        max_q_index=torch.argmax(q_value,dim=1)[0]
        action=max_q_index.detach().item()#récupéré l'index de la qvalue la plus haute
        return action
    
        


class luca():
    #classe luca

    def __init__(self):
        self.nb_action = 0
        self.energy = 0 
        self.color=()
        self.nb_action=0
        self.energy_max=0
        self.age_max=0
        #self.eps=1
        #self.done=False
        self.record_visions = []
        self.record_vision_id=[]


        self.DQN_online=None#à compléter plus tard avec Network
        self.DQN_target=None    
        self.optimizer=None
        self.loss=None
        
        self.avg_reward=[]
        self.stock_avg_rew=[]
        
        
        self.iter_boucle=0
        self.target_q_values=None
        self.max_target_q_values=None
        self.targets=None
        self.q_values=None
        self.action_q_values=None
            
        self.running_loss=0
        
        self.avg_loss_stock=deque(maxlen=1000)
        
    def initialize_dqn(self,st):
        print([np.shape(st)[1],self.nb_action])
        self.DQN_online=Network([np.shape(st)[1],self.nb_action])
        self.DQN_target=Network([np.shape(st)[1],self.nb_action])
        self.optimizer=torch.optim.Adam(self.DQN_online.parameters(), lr=5e-4)


    
    
    def train_dqn(self,replay_buffer,name=None):
        
        """
        fonction qui va gérer l'entrainement
        replay_buffer=matrice qui stock des states (n=BUFFER_SIZE)
        name= juste pour savoir quelle espèce utilise cette fonction
        """
    
        batch_transition=random.sample(replay_buffer,BATCH_SIZE)
        #print(batch_transition[0:12])
        obses=[t[0] for t in batch_transition]
        actions=np.asarray([t[1] for t in batch_transition])
        rews=np.asarray([t[2] for t in batch_transition])
        dones=np.asarray([t[3] for t in batch_transition])
        new_obses=np.asarray([t[4] for t in batch_transition])

        
        
        obses_t=torch.as_tensor(obses,dtype=torch.float32)
        actions_t=torch.as_tensor(actions,dtype=torch.int64).unsqueeze(-1)
        rews_t=torch.as_tensor(rews,dtype=torch.float32).unsqueeze(-1)
        dones_t=torch.as_tensor(dones,dtype=torch.float32).unsqueeze(-1)
        new_obses_t=torch.as_tensor(new_obses,dtype=torch.float32)
        
        
        #--prediction nn (avec target)
        #prediction des qvalues des nouveaux états
        target_q_values=self.DQN_target(new_obses_t)
        #prends la meilleur qvalue de toutes les nouvelles observations (stp1)
        max_target_q_values=target_q_values.max(dim=1,keepdim=True)[0]#[0] car ce .max renvoie un tuple (X,Y) ou X est la maxvalue et Y l'indice de celle-ci
        
        #formule de bellman, calcul des qvalues de st (obses_t) qui serviront de référence
        targets=rews_t+GAMMA*(1-dones_t)*max_target_q_values
        
        #--comparaison avec les q(st) du model_online+ descent de gradient
        #récupère les q values des observations
        q_values=self.DQN_online(obses_t)
        #•prend la qvalue uniquement des actions réellement prises par l'agent
        action_q_values = torch.gather(input=q_values,dim=1,index=actions_t)
        
        self.loss= nn.functional.smooth_l1_loss(action_q_values,targets)

        
        #descente de gradient= optimisation 
        self.optimizer.zero_grad()
        self.loss.backward()#compute le gradient
        self.optimizer.step()#appliquer les gardient
        self.running_loss+=self.loss.item()
        

        if self.iter_boucle % 100 == 0 :#1000 de base
            #print("nombre de jeu",game.nb_game)
            #print('Avg rew',np.mean(rew_buffer))#
            
            
            self.avg_loss_stock.append(self.running_loss/100)
            
            
            print(f"fonction de perte {name} :",self.running_loss)
            #stock_loss.append(avg_loss)
            self.running_loss=0
            self.avg_reward.append(np.mean(rews))
        
        
        
        #update DQN_target
        if self.iter_boucle % TARGET_UPDATE_FREQ ==0:
            #update target_net avec les mêmes caractéristiques que online si condition remplie
            self.DQN_target.load_state_dict(self.DQN_online.state_dict())
            print(f"******avg_rew {name} : ", np.max(self.avg_reward))
            self.stock_avg_rew.append(np.max(self.avg_reward))
            self.avg_reward=[]
            
            
        
        self.iter_boucle+=1
        
    def display_bloc(self,record_posi,color):
        """fonction qui sert à dessiner les obstacles
        record_posi= l'ensemble des positions des individus d'une espèce
        color= couleur des cases
        
        """
        for block in record_posi:
            x_coord=block[0]*CELL_SIZE #essaye de varier cell_size pour voir
            y_coord = block[1] * CELL_SIZE
            block_rect=pygame.Rect(x_coord,y_coord,CELL_SIZE,CELL_SIZE)
            pygame.draw.rect(screen,color,block_rect)
        
    
    def fill_vector_int(max_random,length):
        """
        elle est sensée être très utile, mais je ne l'ai pas utilisé malheureusement. 
        
        max_random=int qui va initialiser la valeur maximale de la variable aléatoire
        length=longueur du vecteur qui sera généré
        """
        return [random.randint(1,max_random) for x in range(1,length)]
    

    def vision(self,positions_head, position_x,position_y, acuity):
        
        """
        fonction qui va générer un champs de vision
        positions_head = position de la tête
        position_x= position de l'individu (x)
        position_y= position de l'individu (y)
        acuity = profondeur du champs de vision (e,g à quel point un individu peut voir loin)
        
        """
        
        #fonction qui servira à gérer les mouvements
        
        #finniiii!! faut juste ajouter la fonction controleur de valeur
        
        if positions_head==2:#haut
            eye = [[None for _ in range(3)] for _ in range(acuity)]
    
            #ola=[2,1,0]
            ola=list(reversed([x for x in range(acuity)]))
            ##print(ola)
            for i in range(acuity) :#pour que eye ne soit pas horizontalement inversé
            
            #x ne varie pas en fct de l'acuité, Y si
                row=[[self.verif_vision(position_x+var_x), self.verif_vision(position_y+ola[i]+1)] for var_x in [-1,0,1]]
                ##print(row)
                eye[i]=row
                ##print([[position_x+var_x, position_y+i+1] for var_x in [-1,0,1]][1])
                #eye=np.array(eye)
        elif positions_head==1:
            eye = [[None for _ in range(acuity)] for _ in range(3)]
            #x varie en fct de l'acuité visuelle, pas Y
            ajout_y=[1,0,-1]#ah ok...
            for i in range(3):#la lecture se fait en y
                row=[[self.verif_vision(position_x+var_x+1), self.verif_vision(position_y+ajout_y[i])] for var_x in range(acuity)]
                eye[i]=row
        elif positions_head==0:#bas
            eye = [[None for _ in range(3)] for _ in range(acuity)]
            ola=[x for x in range(acuity)]
            for i in range(acuity):
                +6#x ne varie en fct de l'acuité, Y si
                row=[[self.verif_vision(position_x+var_x), self.verif_vision(position_y-ola[i]-1)] for var_x in [-1,0,1]]
                eye[i]=row
    
        elif positions_head==3:
            eye = [[None for _ in range(acuity)] for _ in range(3)]
            #x varie en fct de l'acuité, pas y
            ajout_y=[1,0,-1]
            for i in range(3):#la lecture se fait en y
                #x ne varie en fct de l'acuité, Y si
                row=[[self.verif_vision(position_x-var_x-1), self.verif_vision(position_y+ajout_y[i])] for var_x in list(reversed([x for x in range(acuity)]))]
                eye[i]=row
        ##print(eye)
        return eye
    


    def init_progeniture(self,posi,energy=50):
        
        """
        va créer les nouveaux individus
        posi=positions ([x, y])
        """
    
    
        if np.size(posi)==1:
            return
        for baby in posi:
            self.tot_ID+=1

            stat_proge=[baby[0], baby[1], 0,energy, random.randint(0,3), 0, self.tot_ID ]#posi, age=0, energy, posi_tête aléatoire, action=0, ID
            self.record=np.vstack((self.record,stat_proge))
            #on ajoute les 

    def loss_energy_repro(self, ind_loss):
        
        """
        fonction qui va permettre la perte d'énergie pour ceux ayant fait la reproduction
        ind_loss= position des individus ayant fait la repro
        """

        for i in ind_loss:

            self.record_energy[i]-=int(self.energy_max/10) #20 de base/puis 10

    
    def verif_vision(self,x):
        
        """
        fct qui permet de mettre une valeur abberante quand champ de vision excède environnement
        
        """
        #
        if x<0:
            return -10
        else:
            return x
    

    def  init_t0_pop(self,age_max_init, energy_max_init,nb_action,pop_t0=4):
        #position
        coord_pop_x=[random.randint(1,NB_COL) for x in range(1,pop_t0)]
        coord_pop_y=[random.randint(1,NB_ROW) for x in range(1,pop_t0)]
        coord_pop=np.vstack((coord_pop_x,coord_pop_y))
        #age
        age=[random.randint(1,age_max_init) for x in range(1,pop_t0)]
        coord_pop=np.vstack((coord_pop,age))
        #energy
        energy=np.array([energy_max_init for x in range(1,pop_t0) ])
        coord_pop=np.vstack((coord_pop,energy))
        #position(vers où il regarde?)
        position=[random.randint(0,3) for x in range(1,pop_t0)]
        coord_pop=np.vstack((coord_pop,position))
        #actions
        action=[random.randint(0,nb_action-1) for x in range(1,pop_t0)]
        coord_pop=np.vstack((coord_pop,action))
        #ID
        ID=np.array([x for x in range(1,pop_t0) ])
        coord_pop=np.vstack((coord_pop,ID))
        
        ##print("init_fini")
        return coord_pop.transpose()

    
    def mvt_factor(self,collision):
        
        """
        0=rien faire
        1=aller en haut
        2=aller en haut à droite
        3=aller à droite
        4=aller en bas à droite
        5=aller en bas
        6=aller en bas à gauche
        7=aller à gauche
        8=aller en haut à gauche
        9=attaquer
        """
        #if len(self.record_action)!=0:
        ##print("mvt début")
        for row in range(len(self.record_action)):
            
            #x+1
            if self.record_action[row] in [2,3,4] and self.record_positions[row,0]<NB_COL-1 :#(droite/gauche sur pygame)
                if self.is_available(self.record_action[row], self.record_positions[row,:], collision)==True:
                    self.record_positions[row,0]+=1
            
            #x-1
            elif self.record_action[row] in [6,7,8] and self.record_positions[row,0]>0:#(gauche/droite sur pygame)
                if self.is_available(self.record_action[row], self.record_positions[row,:], collision)==True:
                    self.record_positions[row,0]-=1
            
            #y+1
            if self.record_action[row] in [8,1,2] and self.record_positions[row,1]<NB_ROW-1:#(haut/bas sur pygame)
                if self.is_available(self.record_action[row], self.record_positions[row,:], collision)==True:
                    self.record_positions[row,1]+=1
            
            #y-1
            elif self.record_action[row] in [6,5,4] and self.record_positions[row,1]>0:#(bas/haut sur pygame)
                if self.is_available(self.record_action[row], self.record_positions[row,:], collision)==True:
                    self.record_positions[row,1]-=1
            
            #------------orientation tête
            
            elif self.record_action[row]== 9:#vers haut
                self.record_position_head[row]=0
            elif self.record_action[row]== 10:#vers droite
                self.record_position_head[row]=1
            elif self.record_action[row]== 11:#vers bas
                self.record_position_head[row]=2
            elif self.record_action[row]== 12:#vers gauche
                self.record_position_head[row]=3
        ##print("mvt_fin")    
    
    
    def mvt_test(self,collision_plants):
        """
        fonction test
        """
        #fonction test pour tester les mouvements
        for row in range(len(self.record_action)):
            
                if self.record_positions[row,0]>0 and self.record_positions[row,1]<NB_ROW-1:
                    
                    self.record_positions[row,0]-=1
                    self.record_positions[row,1]+=1


    def is_available(self,action, position, collision):
        
        """
        fonction qui va tenter de savoir si une case est disponible
        action= l'action de l'individu
        position= position de l'individu ([x,y])
        collision= les cases qui ne sont pas disponibles
        """
        #initialisation
        posi_x_expected=position[0]
        posi_y_expected=position[1]
        #x+1
        if action in [2,3,4]:
            posi_x_expected=position[0]+1

        #x-1
        elif action in [6,7,8]:
            posi_x_expected=position[0]-1

        #y+1
        if action in [8,1,2]:
            posi_y_expected=position[1]+1

        #y-1
        elif action in [6,5,4]:
            posi_y_expected=position[1]-1

        posi_expected=[posi_x_expected, posi_y_expected]

        if np.any(np.all(collision == posi_expected, axis=1)) == True:
            ##print("ola")
            return False
        ##print("olaaaa")
        return True


    def loss_energy_age(self):
        #fonction qui va gérer les pertes d'énergie 

        """
        gère perte d'énergie... 
        Quand individu fait un mouvement
        Quand individu fait un descendant
        Quand individu se fait attaquer(je ne sais pas encore si ça sera ici.)

        """
        if not self.record.size == 0 :#en vrai ça marchait bien de base, là c'est juste une précaution
            for row in range(np.shape(self.record)[0]):
                if self.record_action[row] in [x for x in range(1,13)]:
                    self.record_energy[row]-=1
            ##print( self.record_energy[1])
                if pygame.time.get_ticks() % 1000 ==0:#à modifier empiriquement malheureusement
                    self.record_age[row]+=1
                    #print("une année en plus")
        ##print("perte energy fin")
        
        
    def manage_dead(self):
        #
        
        """
        fonction qui va gérer la mort
        Quand il fait un mouvement
        Quand il fait un descendant
        Quand il se fait attaquer(je ne sais pas encore si ça sera ici.)
        
        """
        # for row in range(len(np.shape(self.record)[0])):
        #     if self.record_energy[row]<=0:
        #         self.record=np.delete(self.record,row, axis=0)
        
        if not self.record.size == 0 :
            #mort de fatigue
            row_suppr_energy=np.where(self.record_energy<=0)
            self.record=np.delete(self.record,row_suppr_energy, axis=0)
            #mort de veillesse
            row_suppr_age=np.where(self.record_age>=self.age_max)
            self.record=np.delete(self.record,row_suppr_age, axis=0)
        ##print("manage dead fin")


    


class herbiv(luca):
    
    def __init__(self):
        #toutes les infos de tous les individus
        super().__init__()#heritage de luca
        #on modifie certaines variables qui viennent de l'héritage + en ajoute d'autres
        self.color=(10,200,50)
        self.nb_action=12
        self.energy_max=7000#200 de base/puis 1200/et 3000
        self.age_max=400

        #NP_pop=35
        self.record=self.init_t0_pop(100, self.energy_max,self.nb_action,100 )#nombre d'éléments, qui seront les valeurs input du nn (positions, energie, etc.)
        
        
        #self.luca_her=luca()
        self.tot_ID=np.shape(self.record)[0]

        
        
        self.record_visions = []
        self.record_vision_id=[]
        self.determ_vision()
        


    #------------------------délimitation record
    @property#decorateur pour lier record_positions et record
    def record_positions(self):
        return self.record[:,0:2]
    
    @record_positions.setter #pour faire en sorte qu'une modif de record_posi soit une modif de record
    def record_positions(self, new_values):
        self.record[:,0:2]=new_values

    #age
    @property
    def record_age(self):
        return self.record[:,2]
    
    @record_age.setter
    def record_age(self, new_values):
        self.record[:,2]=new_values
        
    #energy
    @property
    def record_energy(self):
        return self.record[:,3]
    
    
    @record_energy.setter
    def record_energy(self, new_values):
        self.record[:, 3] = new_values
    #position_head
    @property
    def record_position_head(self):
        return self.record[:,4]
    
    @record_position_head.setter
    def record_position_head(self, new_values):
        self.record[:, 4] = new_values
    
    @property
    def record_action(self):
        return self.record[:,5]
    
    @record_action.setter
    def record_action(self, new_values):
        self.record[:, 5] = new_values
    
    @property
    def record_ID(self):
        return self.record[:,6]
    
    @record_ID.setter
    def record_ID(self, new_values):
        self.record[:, 6] = new_values
    
        
    def determ_vision(self):
        """
        fonction qui sert à initialiser le vecteur de champs de vision (champs de vision pour chaque individus)
        """
        #va permettre de reremplir la matrice des visions
        self.record_visions = []
        self.record_vision_id=[]
        for row in self.record:
            #vision = luca.vision(self.record_position_head[row],self.record_positions[row,0], self.record_positions[row,1], 4)
            vision = self.vision(row[4],row[0], row[1], 4)
            self.record_visions.append(vision)
            self.record_vision_id.append(row[6])
            #gere ça gros
        #self.record_visions=list(reversed(self.record_visions))
        #self.record_vision_id=self.record_vision_id
        
        
    

    
class carne(luca):
    def __init__(self):
        #toutes les infos de tous les individus
        super().__init__()
        self.color=(200,20,50)
        self.nb_action=12
        self.energy_max=6700#100 de base/puis 900/et 2700
        self.age_max=300
        self.record=self.init_t0_pop(self.age_max, self.energy_max,self.nb_action,15)#12
        self.tot_ID=np.shape(self.record)[0]
        
        self.record_visions=[]
        self.record_vision_id=[]
        self.determ_vision()
        
    
    #0/1 = position
    @property#decorateur pour lier record_positions et record
    def record_positions(self):
        return self.record[:,0:2]
    
    @record_positions.setter #pour faire en sorte qu'une modif de record_posi soit une modif de record
    def record_positions(self, new_values):
        self.record[:,0:2]=new_values

    # 2 =age
    @property
    def record_age(self):
        return self.record[:,2]
    
    @record_age.setter
    def record_age(self, new_values):
        self.record[:,2]=new_values
        
    # 3 =energy
    @property
    def record_energy(self):
        return self.record[:,3]
    
    
    @record_energy.setter
    def record_energy(self, new_values):
        self.record[:, 3] = new_values
        
        
    # 4 =position_head
    @property
    def record_position_head(self):
        return self.record[:,4]
    
    @record_position_head.setter
    def record_position_head(self, new_values):
        self.record[:, 4] = new_values
    
    # 5 =action
    @property
    def record_action(self):
        return self.record[:,5]
    
    @record_action.setter
    def record_action(self, new_values):
        self.record[:, 5] = new_values
    
    # 6 = ID
    @property
    def record_ID(self):
        return self.record[:,6]
    
    @record_ID.setter
    def record_ID(self, new_values):
        self.record[:, 6] = new_values
        

    def determ_vision(self):
        """
        fonction qui sert à initialiser le vecteur de champs de vision (champs de vision pour chaque individus)
        """
        #va permettre de reremplir la matrice des visions (à chaque pas de temps)

        self.record_visions = []
        self.record_vision_id=[]
        for row in self.record:
            #vision = luca.vision(self.record_position_head[row],self.record_positions[row,0], self.record_positions[row,1], 4)
            vision = self.vision(row[4],row[0], row[1], 3)
            self.record_visions.append(vision)
            self.record_vision_id.append(row[6])
        #self.record_visions=list(reversed(self.record_visions))
        #self.record_vision_id=list(reversed(self.record_vision_id))
    

    
    
            
class plants():
    def __init__(self):
        #toutes les infos de tous les individus
        self.color=(82,177,253)
        self.energy_max=2000#puis 3000/et puis 4000/
        self.age_max=1000
        self.record=plants.init_t0_pop(self.age_max, self.energy_max,500)
        self.tot_ID= np.shape(self.record)[0]
        self.iter_activit_repro=0
        
        


        
    #---position
    @property
    def record_positions(self):
        return self.record[:,0:2]
    
    @record_positions.setter
    def record_positions(self, new_values):
        self.record[:,0:2] = new_values
    
    #----age
    @property
    def record_age(self):
        return self.record[:,2]
    
    @record_age.setter
    def record_age(self, new_values):
        self.record[:, 2] = new_values
    
    #----energy
    @property
    def record_energy(self):
        return self.record[:,3]
    
    @record_energy.setter
    def record_energy(self, new_values):
        self.record[:, 3] = new_values
    
    @property
    def record_ID(self):
        return self.record[:,4]
    
    @record_ID.setter
    def record_ID(self, new_values):
        self.record[:, 4] = new_values
        
        
    
    def age_function(self,age_switch,input_age,min_max_list_energy,energy_t0,list_age):
        
        """
        fonction inutile pour le moment(oui, j'ai le démon')
        
        age_switch : age à partir duquel la tendance va vers le bas
        
        input_age : l'age qui entre en input
        
        min_max_list_energy : liste qui va choper l'energie maximum (avant le switch) et minimum (après le switch)
        
        energy_t0 : energy au t0
        
        age_t0
        
        list_age : list qui contient les ages qui vont cadriller le niveau d'energie  [age_t0, age_extremum_1, age_extremum_2]
        
        """
        
        def function_up(x,age_0):
            return 1.2*x+age_0
        
        def function_down(x,age_t0):
            return -0.5*x+ age_t0
        
        if input_age==list_age[0]:
            energy=energy_t0
        elif input_age>=list_age[1] and input_age<=age_switch:
            pass
            #energy=
        elif input_age<=age_switch:
            energy=function_up(input_age,list_age[0])
        elif input_age>age_switch:
            energy=function_down(input_age,function_up(age_switch,list_age[1]))
        else:
            #print("y a un truc qui ne va pas")
            quit
        

    def init_t0_pop(age_max_init, energy_max_init,pop_t0=40):
        
        """
        fonction qui va initialiser les individus au temps initial
        age_max_init= l'age maximale de chaque individu
        energy_max_init = energie maximale
        pop_t0=nombre d'individu à t0
        
        """
        #position
        coord_pop_x=[random.randint(1,NB_COL) for x in range(1,pop_t0)]
        coord_pop_y=[random.randint(1,NB_ROW) for x in range(1,pop_t0)]
        coord_pop=np.vstack((coord_pop_x,coord_pop_y))
        
        for i in range(np.shape(coord_pop)[0]):
            if len(np.where(np.all(coord_pop == coord_pop[i]))) >=2:
                coord_pop=np.delete(coord_pop, i, 1)
                loop_gene=True
                while loop_gene:
                    coord_edit=[random.randint(1,NB_COL),random.randint(1,NB_ROW)]
                    if len(np.where(np.all(coord_pop == coord_pop[i]))) >=2:
                        continue
                    else:
                        np.hstack((coord_pop, coord_edit))
                        loop_gene=False
        #verifier lignes uniques
        lignes_uniques, indices, inverse, counts = np.unique(coord_pop, axis=0, return_index=True, return_inverse=True, return_counts=True)

        ##print("shape",np.shape(coord_pop))
        ##print(lignes_uniques[counts>1])
        #
        #age
        age=[random.randint(1,age_max_init) for x in range(1,pop_t0)]
        coord_pop=np.vstack((coord_pop,age))
        #init_age=np.array([plants.age_function(30,x, [20,50],[1,100]) for x in age ])
        #energy
        energy=np.array([energy_max_init for x in range(1,pop_t0) ])
        coord_pop=np.vstack((coord_pop,energy))
        
        ID=np.array([x for x in range(1,pop_t0) ])
        coord_pop=np.vstack((coord_pop,ID))
        
        #tout_pop
        return coord_pop.transpose()
    
        
    def init_progeniture(self,posi,energy=40):
        
        """
        fonction qui va gérer la création des progénitures
        posi= position([x,y])
        """
        
        # if np.size(posi)==1:
        #     #print("probleme")
        #     return
        
        if posi==None:
            return
        #print(posi)
        for baby in posi:
            self.tot_ID+=1
 
            stat_proge=[baby[0], baby[1], 0,energy, self.tot_ID ]#posi, age=0, energy, posi_tête aléatoire, action=0, ID
            self.record=np.vstack((self.record,stat_proge))
        
    # def repro_plants(self):
    #     stock_capable_repro=np.where(self.record_age>3 and self.record_age <100)
        
        #stock
        
class sim_evol():
    #classe chaperonne qui va gérer la simulatiob
    def __init__(self):
        self.carne=carne()
        self.plants=plants()
        self.luca=luca()
        
        
        self.herbiv=herbiv()
        self.view_vision=False
        
        #j'en ai besoin je le craint pour les repro
        self.carne_where_repro=[]
        self.herbiv_where_repro=[]
        
        
        
        #self.DQN_carne=luca.DQN()
    
    def repro_animal(self,data_animal):
        #reproduction des animaux
        lignes_uniques, indices, inverse, counts = np.unique(data_animal, axis=0, return_index=True, return_inverse=True, return_counts=True)
        ##print(inverse)
        doublons=lignes_uniques[counts > 1]
        if len(doublons)>0:
            #print("doublons : ",doublons)

            novo=[]#les nouveaux nées
            where_repro=[]#pour enlever les points de vie
            for doub_iter in doublons:
                ##print("doub_iter:", doub_iter)
                for_return=False
                for i in [-1,0,1]:
                    if for_return:#permet  de revenir à la première boucle
                        break
                    for j in [-1,0,1]:
                        #si dépasse les zones du jeu, passe
                        if doub_iter[0]+i>=NB_COL or doub_iter[0]+i<0 or doub_iter[1]+j >=NB_ROW or doub_iter[1]+j <=0:
                            ##print("non la t'abuses")
                            continue
                        
                        
                        if [doub_iter[0]+i,doub_iter[1]+j] not in self.herbiv.record_positions.tolist()  and  [doub_iter[0]+i,doub_iter[1]+j] not in novo: #and [doub_iter[0]+i,doub_iter[1]+j] not in self.carne.record_positions.tolist():
                            novo.append([doub_iter[0]+i,doub_iter[1]+j])
                            #perte d'énergie
                            where_repro.append(np.where(np.all(data_animal == doub_iter, axis=1))[0])
                            
                            for_return=True#permet de revenir à la première boucle
                            break
            
            return [novo,where_repro]#,  where_repro
            ##print("doublons:" ,doublons)
        
    
        
        
        
    def bouff_carne(self):
        def miam(self):
            print("miam")
            self.herbiv.record_energy[row_herbiv]-=self.herbiv.energy_max/4#herbiv qui prend un coup
            if self.herbiv.record_energy[row_herbiv]<=0:#si l'herbiv meurt sur le coup, le carniv gagne de l'énergie
                self.carne.record_energy[row_carne]+=self.carne.energy_max/4
            
        #is_eating=False
        for row_herbiv in range(np.shape(self.herbiv.record)[0]):
            for row_carne in range(np.shape(self.carne.record)[0]):
                if self.carne.record_action[row_carne]==12:
                    #c'est débile là, tu sais?
                    if self.carne.record_position_head[row_carne]==0 and self.herbiv.record_positions[row_herbiv].tolist() in self.carne.record_visions[row_carne][0]:
                        miam()
                    elif self.carne.record_position_head[row_carne]==1 and self.herbiv.record_positions[row_herbiv].tolist() in [self.carne.record_visions[row_carne][w][0] for w in range(3) ]:
                        miam()
                    elif self.carne.record_position_head[row_carne]==2 and self.herbiv.record_positions[row_herbiv].tolist() in self.carne.record_visions[row_carne][-1]:
                        miam()
                    elif self.carne.record_position_head[row_carne]==2 and self.herbiv.record_positions[row_herbiv].tolist() in [self.carne.record_visions[row_carne][w][-1] for w in range(3) ]:
                        miam()
                    


                
    def repro_plants(self):
        if self.plants.iter_activit_repro <50:#permet d'endiguer la reproduction
            self.plants.iter_activit_repro +=1
            return
        novo=[]
        self.plants.iter_activit_repro = 0
        for iter_plants in range(len(self.plants.record_age)):
            #print("age", self.plants.record_age[iter_plants])
            if self.plants.record_age[iter_plants]<5 and  self.plants.record_age[iter_plants]>900:
                action_repro=random.randint(5, 1000)
            else:
                action_repro=random.randint(50, 1000)
            #print("action_repro", action_repro, "ID NB ", self.plants.record_ID[iter_plants])
            if action_repro > 900:
                posi_iter=self.plants.record_positions[iter_plants]
                #print("posi_iter", posi_iter)
                ##print("posi_iter", posi_iter)
                #repro
                
                for_return=False
                for i in [-1,0,1]:
                    if for_return:#permet  de quitter la première boucle car on a trouvé les coordonnées
                        break
                        #pass
                    for j in [-1,0,1]:
                        
                        if [posi_iter[0]+i,posi_iter[1]+j] not in self.herbiv.record_positions.tolist() and  [posi_iter[0]+i,posi_iter[1]+j] not in self.plants.record_positions.tolist() :#and [posi_iter[0]+i,posi_iter[1]+j] not in self.carne.record_positions:
                            ##print("trouvé")
                            novo.append([posi_iter[0]+i,posi_iter[1]+j])
                            for_return=True#permet de revenir à la première boucle
                            break
        ##print("novo_plants", novo)
        ##print("fin repro")    
        return novo
        

        
    def draw_simu_element(self):
        """fonction qui va dessiner les différents éléments de la simulation"""
        ##print((np.shape(self.plants.record_positions)))
        self.luca.display_bloc(self.herbiv.record_positions,self.herbiv.color)
        self.luca.display_bloc(self.plants.record_positions,self.plants.color)
        self.luca.display_bloc(self.carne.record_positions,self.carne.color)
        
        if self.view_vision:
            if self.carne.record_vision_id !=[]:
                ##print(self.carne.record_visions)
                for i in self.carne.record_visions :
                    #permet d'afficher le champ de vision de carne
                    ##print(i)
                    [self.luca.display_bloc( defile , (130,130,250)) for defile in i]
        
    
    
    def bouff_herbi(self):#modifié pour ajouter if action=12
        
        """
        fonction qui va servir à gérer la sustentation pour herbiv
        
        """
    
        for row in self.herbiv.record:
            ##print("posi", row[0:2])
            if row[5]==11:
                is_eating=np.where(np.all(self.plants.record_positions==row[0:2], axis=1))
    
    
                if len(is_eating[0]) > 0:
                    ##print("is_eating : ", is_eating)
                    is_eating=is_eating[0]
                    ##print("is_eating_2 : ", is_eating)
                    
                    ##print()
                    for eat in is_eating:
                        # #print("eat : ",eat)
                        # #donne energie
                        # #for 
                        # #print("energy before",row[3])
                        row[3]+=int(self.plants.record_energy[eat]/2)#bon, il y a un soucis genre eat est out of bound
                        # #print("energy after",row[3])
                        # #print("----------")
                        #tuer plants
                        self.plants.record=np.delete(self.plants.record, eat, axis=0)
                        ##print(np.shape(self.plants.record))
                
                
    
    def restart(self):
        """
        redémarrage simulation
        """
        #ppur redémarer 
        self.carne=carne()
        self.plants=plants()
        self.luca=luca()
        
        
        self.herbiv=herbiv()
        self.view_vision=False



    
    def check_and_update_simu(self, auto_herbiv=True, auto_carne=True):
        
        """
        fonction centrale qui va permettre à la simulation d'avancer d'un pas de temps
        
        auto_herbiv/auto_carne= booléen qui va permettre de gérer l'automatisation de la prise d'action (est-ce strictement aléatoire ou est-ce que le DQN peut intervenir?)
        """
        

        #-------choix action
        #herbiv
        if auto_herbiv:
            take_action(self.herbiv,self.herbiv.nb_action)
        elif auto_herbiv==False:
            
            iter_eps_herbiv=self.herbiv.iter_boucle
            if iter_eps_herbiv>=len(eps_linsp)-1:
                iter_eps_herbiv=len(eps_linsp)-1
                
            #print("eps_linsp[iter_eps_herbiv] : ", eps_linsp[iter_eps_herbiv])

            take_action(self.herbiv,self.herbiv.nb_action,eps=eps_linsp[iter_eps_herbiv],DQN=self.herbiv.DQN_online,st=st_herbiv)

        #carne
        if auto_carne:
            take_action(self.carne,self.carne.nb_action)
            
        elif auto_carne==False:
            iter_eps_carne=self.carne.iter_boucle
            if iter_eps_carne>=len(eps_linsp)-1:
                iter_eps_carne=len(eps_linsp)-1
            
            #print("eps_linsp[iter_eps_carne] : ", eps_linsp[iter_eps_carne])
            take_action(self.carne,self.carne.nb_action,eps=eps_linsp[iter_eps_carne],DQN=self.carne.DQN_online,st=st_carne)
            
            
        self.herbiv_act=self.herbiv.record_action
        self.carne_act=self.carne.record_action
            
        
        ##print(self.plants.record_positions)
        
        #--------faire action
        #self.herbiv.mvt_factor(np.array([[0,0],[0,0]]))#je dois mettre les posis des carnivors 
        self.herbiv.mvt_factor(self.carne.record_positions)
        self.carne.mvt_factor(self.herbiv.record_positions)
        
        
        #self.herb
        
        #self.herbiv.mvt_test(self.plants.record_positions)
        #self.herbiv.luca_her.mvt_test(self.plants.record_positions)
        #self.simu_herattack()
        #---------
        self.herbiv.determ_vision()
        self.carne.determ_vision()
        

        
        
        #--miam miam
        self.bouff_herbi()
        self.bouff_carne()
        
        #--------repro
        #herbiv
        stock_novo_herb =self.repro_animal(self.herbiv.record_positions)
        
        # #print(stock_loss_energy)
        ##print("herbiv stock repro : ",stock_novo_herb)
        if stock_novo_herb != None:
            self.herbiv.init_progeniture(stock_novo_herb[0])
            self.herbiv_where_repro=stock_novo_herb[1]
            self.herbiv.loss_energy_repro(stock_novo_herb[1])
            #pour éviter certains bugs lors de la creation de stp1
            self.herbiv.determ_vision()
            
         #carne
        stock_novo_carne=self.repro_animal(self.carne.record_positions)
        ##print("carniv stock repro : ",stock_novo_carne)
        if stock_novo_carne != None:
            ##print("carniv stock repro[0] : ",stock_novo_carne)
            self.carne.init_progeniture(stock_novo_carne[0])
            self.carne_where_repro=stock_novo_carne[1]
            self.carne.loss_energy_repro(stock_novo_carne[1])
            #pour éviter certains bugs lors de la creation de stp1
            self.carne.determ_vision()
         #plants
        
        if np.shape(self.plants.record_ID)[0]<1830 and np.shape(self.herbiv.record_ID)[0]>15:#si je ne veux pas que mon ordi brule
            stock_novo_plants=self.repro_plants()
            ##print("stock_novo_plants",stock_novo_plants)
            self.plants.init_progeniture(stock_novo_plants)
        
        
        #self.repro_plants()
        
                
        #--------aging and death
        #herbi
        self.herbiv.loss_energy_age()
        self.herbiv.manage_dead()
        #carne
        self.carne.loss_energy_age()
        self.carne.manage_dead()
        
        
        # if self.herbiv.tot_ID >= 80:
        #     #print("test restart!!!")
        #     self.restart()

def plot_stat(donnee,name):
    
    """
    va plot des tableaux
    donnée= les données
    name= l'espèce'
    
    """
    print(f"donnée {name} : ", donnee)
    print(f"len donnée {name}", len(donnee))
    print(f"abscisse {name}", np.linspace(0,len(donnee),len(donnee)))
    print(f"len abscisse {name}", len(np.linspace(0,len(donnee),len(donnee))))
    
    plt.plot([x for x in range(len(donnee))],donnee,label=f"plot pour {name}")
    plt.legend()


pygame.init()  # initialisation des modules

# taille écran
#10/15
# NB_COL = int(80*1.5)
# NB_ROW = int(40*1.5)
NB_COL = 80
NB_ROW = 40

CELL_SIZE = 10#normalement 20

timer = pygame.time.Clock()



#---DQN stuff
BUFFER_SIZE=25000#taille du buffer
BATCH_SIZE=200#taille du batch (nb éléments qui seront pris dans le buffer pour entrainer le réseau de neurone)
#MIN_REPLAY_SIZE=1000#replay minimum avant l'entrainement principal
replay_buffer_carne = deque(maxlen=BUFFER_SIZE)
replay_buffer_herbiv = deque(maxlen=BUFFER_SIZE)
TARGET_UPDATE_FREQ=1000#fréquence où le 2ème NN va être réinitialisé
eps_start=0.7#epsilon-greedy début
#eps_end=0.02
eps_end=0.08#epsilon-greedy fin (0.08 et pas moins car je veux encore que mon modèle continue de tester)
GAMMA=0.7#valeur qui va définir l'importance que l'agent va accorder aux récompenses futures par rapport aux récompenses immédiates#O.5 de base
eps=1#eps pour l'étape initial
fluctu_remem=0#retenir la vitesse, servira pour l'état
#---
simul_chaine=sim_evol()

##print(simul_chaine.plants.record)

screen = pygame.display.set_mode(size=(NB_COL * CELL_SIZE, NB_ROW * CELL_SIZE))
pygame.display.set_caption("survivor")#titre fenêtre

# prendre en compte les touches du joueur
SCREEN_UPDATE = pygame.USEREVENT
pygame.time.set_timer(SCREEN_UPDATE, 100)  # utilisé pour créer un timer de 200millisec.


st_herbiv=[ [*simul_chaine.herbiv.record[i,:5], *vision_to_pixel( simul_chaine.herbiv.record_visions[i])] for i in range(np.shape(simul_chaine.herbiv.record)[0]) ]
st_carne=[ [*simul_chaine.carne.record[i,:5], *vision_to_pixel( simul_chaine.carne.record_visions[i])] for i in range(np.shape(simul_chaine.carne.record)[0]) ]


#DQN
simul_chaine.carne.initialize_dqn(st_carne)
simul_chaine.herbiv.initialize_dqn(st_herbiv)


#st_carne=[]
game_on = True
pygame.time.get_ticks()
#eps=variable global


iter_boucle=0
BEGIN_TRAIN=20000#4000/puis 12000

eps_linsp=np.linspace(eps_start,eps_end,3000)#1000 de base

while game_on:
    # pour quitter
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
           
            plot_stat(simul_chaine.herbiv.avg_loss_stock,"herbiv")
            plot_stat(simul_chaine.carne.avg_loss_stock,"carne")
            del simul_chaine #pour éviter les problems car parfois quand tu cut avec le carré rouge, les valeurs restent
            pygame.quit()
            sys.exit()
        if event.type == SCREEN_UPDATE:
            if np.shape(simul_chaine.herbiv.record)[0] == 0 and np.shape(simul_chaine.herbiv.record)[0] == 0:
                #ola plot
                plot_stat(simul_chaine.herbiv.avg_loss_stock,"herbiv")
                plot_stat(simul_chaine.carne.avg_loss_stock,"carne")
                
                pygame.quit()
                sys.exit()
                simul_chaine.restart() #on relance tout
                st_herbiv=[ [*simul_chaine.herbiv.record[i,:5], *vision_to_pixel( simul_chaine.herbiv.record_visions[i])] for i in range(np.shape(simul_chaine.herbiv.record)[0]) ]
                st_carne=[ [*simul_chaine.carne.record[i,:5], *vision_to_pixel( simul_chaine.carne.record_visions[i])] for i in range(np.shape(simul_chaine.carne.record)[0]) ]

            
            #done et reward
            done=[]
            reward=[]
            #recup ID pour les comparer avec la step +1
            

            id_herbiv=simul_chaine.herbiv.record_ID
            id_carne=simul_chaine.carne.record_ID
            #
            
            HP_herbiv_past=simul_chaine.herbiv.record_energy
            HP_carne_past=simul_chaine.carne.record_energy
            

            
            #gestion automatique
            
            #juste pour savoir quand l'entrainement commence
            warn_training(len([1 for i in replay_buffer_herbiv]), "herbivores")
            warn_training(len([1 for i in replay_buffer_carne]), "carnivores")
            
            if len([1 for i in replay_buffer_herbiv]) >= BEGIN_TRAIN:
                auto_herbiv=False
            else:
                auto_herbiv=True
            
            if len([1 for i in replay_buffer_carne]) >= BEGIN_TRAIN:
                auto_carne=False
            else:
                auto_carne=True
                
            simul_chaine.check_and_update_simu(auto_herbiv,auto_carne)
            
            #on doit pas considérer les ex nihilo pour les stp1!!
            
            #on ne veut pas mettre dans stp1 les nouveaux_nées
            id_stp1_herbiv=np.where(np.isin(simul_chaine.herbiv.record_ID,id_herbiv))[0]
            stp1_herbiv=[ [*simul_chaine.herbiv.record[i,:5], *vision_to_pixel( simul_chaine.herbiv.record_visions[i])] for i in id_stp1_herbiv.tolist() ]

            
            #juste des indices ici hein!
            ind_id_stp1_carne=np.where(np.isin(simul_chaine.carne.record_ID,id_carne))[0]
            true_id_stp1_carne=simul_chaine.carne.record_ID[ind_id_stp1_carne]
            stp1_carne=[ [*simul_chaine.carne.record[i,:5], *vision_to_pixel( simul_chaine.carne.record_visions[i])] for i in ind_id_stp1_carne.tolist() ]
                        
            
            #remplissage du replay buffer
            if stp1_herbiv!=[]:
                for i in range(np.shape(id_herbiv)[0]):
                    replay_buffer_herbiv.append(
                        [st_herbiv[i],#st
                        simul_chaine.herbiv_act[i],#action
                        reward_done(id_herbiv[i],simul_chaine.herbiv, simul_chaine.herbiv_where_repro, HP_herbiv_past[i])[0],#reward
                        reward_done(id_herbiv[i],simul_chaine.herbiv, simul_chaine.herbiv_where_repro, HP_herbiv_past[i])[1],#done
                        id_stp1_transi_manage(id_herbiv[i], simul_chaine.herbiv,stp1_herbiv)]) #stp1
                     
                    
            if stp1_carne!=[]:
                for i in range(np.shape(id_carne)[0]):
                    replay_buffer_carne.append(
                        [st_carne[i],#st
                        simul_chaine.carne_act[i],#action
                        reward_done(id_carne[i],simul_chaine.carne, simul_chaine.carne_where_repro, HP_carne_past[i])[0],#reward
                        reward_done(id_carne[i],simul_chaine.carne, simul_chaine.carne_where_repro, HP_carne_past[i])[1],#done
                        id_stp1_transi_manage(id_carne[i], simul_chaine.carne,stp1_carne)]) #stp1
            


            
            
            if len([1 for i in replay_buffer_herbiv]) >= BEGIN_TRAIN:
                simul_chaine.herbiv.train_dqn(replay_buffer_herbiv,"herbiv")
                
            if len([1 for i in replay_buffer_carne]) >= BEGIN_TRAIN:
                simul_chaine.carne.train_dqn(replay_buffer_carne,"carne")
                

            
            
            
            #c'est st ça!
            st_herbiv=[ [*simul_chaine.herbiv.record[i,:5], *vision_to_pixel( simul_chaine.herbiv.record_visions[i])] for i in range(np.shape(simul_chaine.herbiv.record)[0]) ]
            st_carne=[ [*simul_chaine.carne.record[i,:5], *vision_to_pixel( simul_chaine.carne.record_visions[i])] for i in range(np.shape(simul_chaine.carne.record)[0]) ]

            #ne fait plus ça si tu ne veux pas de problèmes: stp1 ne prends pas en compte les nouveaux nés!
            #st_herbiv=stp1_herbiv
            #st_carne=stp1_carne
            


        

    screen.fill(pygame.Color('white')) 
    simul_chaine.draw_simu_element()
    show_grid()
    ##print(pygame.time.get_ticks())
    pygame.display.update()  # permettre la mise à jour des données à chaque pas de temps
    timer.tick(200)  # "fps"
    pygame.display.flip()
    

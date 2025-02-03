"""
date = 03/02/2025

@author: asyl_scy (Kara)

topic= On va apprendre à un réseau de neurone à jouer à mon jeu!!

Il s'agit ici de l'étape d'entrainement du DQN qui apprendra à jouer à mon jeu "save the cat", disponible dans my_games/tes_jeux_cube_game_feature2.3.py .
"""
import pygame
from pygame.locals import*
import sys
import random
import numpy as np
from random import randint
from collections import deque

from torch import nn
import torch
import itertools



"""
On va tenter de faire du DQN avec le jeux cube game (save the cat)
action=["droite", "gauche", "dash"] = [0,1,2]

"""



def save_model(model, name):
    """
    model=le modèle qui sera exporté
    name=bout de nom de fichier (qui est le nom du modèle)
    """
    #fonction qui va servir à sauvegarder les poids etc du réseau de neurone.

    PATH = './'+name+'_asyl.pth'
    torch.save(model.state_dict(), PATH)

def show_grid():
    """sert à montrer la grille"""
    for i in range(0, NB_COL):
        for j in range(0, NB_ROW):
            rect = pygame.Rect(i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, pygame.Color("black"), rect, width=1)

def score_musique (score):#relancer la musique si on pert
    """sert à remettre la musique en cas de perte """
    if score==0:
        pygame.mixer.music.play(-1, 0.0, 0)

def high_s(score, highscore):
    """sert a configurer le meilleur score
    score= le score
    highscore=le score le plus haut
    
    """
    if score> highscore:
        return score
    else:
        return highscore


class Network(nn.Module):
    
    """
    classe réseau de neurone (pytorch)
    """
    def __init__(self,info_transition):
        
        #info_transition[0]=len(st), info_transition[1]=len(action)
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(info_transition[0], 64),
            nn.Tanh(),#bah mince, je voulais mettre relu
            nn.Linear(64,info_transition[1])
            )
        
    def forward(self,x):#si pas présent, bug
        return self.net(x)
    
    def act(self,st):

        """
        fonction qui va retournée la meilleure action
        st=l'état.
        """
        
        #récupère l'action avec la meilleur q_value
        st_t=torch.as_tensor(st,dtype=torch.float32)#torch tensor
        q_value=self(st_t.unsqueeze(0))#t'ajoute un plan
        #on prend l'action avec la meilleure q_value
        max_q_index=torch.argmax(q_value,dim=1)[0]
        action=max_q_index.detach().item()#récupéré l'index de la qvalue la plus haute
        return action
    
        
def take_action(eps,DQN=None,st=None):#step, transition, eps (epsilon greedy)

    """
    eps=epsilon-greedy
    DQN=le modèle qui pourrait être utilisé.
    st=l'état
    """
    if random.uniform(0, 1) < eps:
        action = randint(0, 2)
    else: # Or greedy action
        action = DQN.act(st)#selectionne action qui maximise l'espérance à l'état st 
    return action

class Block:
    
    """
    classe qui permettra à créer des blocks
    """
    # sert à dessiner les blocs (utile pour la phase "3")
    def __init__(self, x_pos, y_pos):
        self.x = x_pos
        self.y = y_pos


class player:
    """la classe du joueur"""
    def __init__(self):
        # position en x/y
        self.x = 3
        self.y = 3
        self.block = Block(self.x, self.y)
        self.direction = "DOWN"
        
    def draw_player(self):
        """fonction qui sert à dessiner le bloc du joueur"""
        rect = pygame.Rect(self.block.x * CELL_SIZE, self.block.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, (72, 212, 98), rect)  # lieu ou il sera dessiné, colo et ce qu'on dessine

    def mvt_player(self):
        """fonction qui sert à initialiser les mouvements du joueur"""
        varia=1        # conséquence de la touche clavier sur le joueur
        # quoi qu'il se passe, on met y+1, le déplacement diagonal est plus fluide que le déplacement horizontal puis vertical (source=moi)
        if self.direction == "DOWN":
            self.block = Block(self.block.x, self.block.y + varia)
        if self.direction == "RIGHT":
            self.block = Block(self.block.x + varia, self.block.y + varia)
        if self.direction == "LEFT":
            self.block = Block(self.block.x - varia, self.block.y + varia)
        if self.direction == "DASH":
            self.block = Block(self.block.x , NB_ROW-varia)

        # quoi qu'il se passe, ça finit par descendre
        self.direction = "DOWN"


    def nextlvl_player(self):
        """position du joueur au niveau suivant"""
        x = random.randint(int(NB_COL*0.33), int(NB_COL*0.66))
        y = 3
        self.block = Block(x, y)


    def reroll_player(self):
        """position du joueur après une défaite"""
        x = 3
        y = 3
        self.block = Block(x, y)




class goal():
    """classe de l'endroit où doit aller le joueur (goal)"""
    def __init__(self):
        # position en x/y
        self.x = NB_COL/2
        self.y = NB_ROW-0.2
        self.block = Block(self.x, self.y)

    def draw_goal(self):
        """fonction qui sert à dessiner le bloc du goal"""
        rect = pygame.Rect(self.block.x * CELL_SIZE, self.block.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, (0,0,0), rect)



    def reset_goal(self):
        """fonction qui sert à réinitialiser le goal (après une défaite du joueur)"""
        self.x = 3
        self.y = NB_ROW - 0.2
        self.block = Block(self.x, self.y)
        return self.x,self.y

    # on change juste les coordonnées quand on réussi un niveau
    def nextlevel_goal(self):
        """fonction qui sert à positionner le goal après un succès du joueur"""
        self.x = random.randint(int(NB_COL*0.33), int(NB_COL*0.66))
        self.y = NB_ROW - 0.2
        self.block = Block(self.x, self.y)





class obstacle():
    """classe des obstacles (les toucher provoque un game over)"""
    def __init__(self,a ,b ,a1,b1,a2,b2):
        # position en x/y
        self.body=[Block(a,b),Block(a1,b1),Block(a2,b2)]#les positions des trois blocs (peut mieux faire en vrai)
        self.attention=(6,7,8,9,16,17,18)#les moments (scores) où il y aura les pièges

    def draw_obstacle(self):
        """fonction qui sert à dessiner les obstacles"""
        for block in self.body:
            x_coord=block.x*CELL_SIZE #essaye de varier cell_size pour voir
            y_coord = block.y * CELL_SIZE
            block_rect=pygame.Rect(x_coord,y_coord,CELL_SIZE,CELL_SIZE)
            pygame.draw.rect(screen,(82,177,253),block_rect)






class game():
    """class qui va géré le jeu"""
    def __init__(self):
        self.player=player()
        self.goal= goal()
        self.obstacle=obstacle(2,6,5,6,8,6)
        self.mark_obstacle=False
        self.score=0
        self.highscore=0
        self.nb_game=0
        self.done=False

    def update(self):
        """fonction qui va considérer les nouvelles informations"""
        self.player.mvt_player()#pour avoir la position du joueur
        self.check_player_on_goal()
        self.game_over()

    def reset(self):
            """fonction qui sert à tout recommencer à 0"""
        #tout recommencer à 0, sert utile pour le main entrainement
            self.player.reroll_player()
            self.goal.reset_goal()
            self.highscore=0
            self.score=0
        
        
    def draw_game_element(self):
        """fonction qui va dessiner le joueur et goal"""
        self.player.draw_player()
        self.goal.draw_goal()
        for i in self.obstacle.attention:
            if self.score ==i:
                self.obstacle.draw_obstacle()


    def check_player_on_goal(self):
        """fonction qui va gérer les cas où le joueur a réussi à toucher la cible (goal)"""
        player_block = self.player.block
        goal_block= self.goal.block
        if player_block.x == goal_block.x and player_block.y == NB_ROW - 1:
            self.player.nextlvl_player()
            if  self.score>7:
                self.goal.nextlevel_goal()
            self.score+=1
            print(self.score)
        #t'as pas mis le else

    def game_over(self):# correct
        """fonction qui va gérer les cas de game over"""
        player_block = self.player.block
        
        if player_block.x not in range(0, NB_COL) or player_block.y not in range(0, NB_ROW):
            self.player.reroll_player()
            self.goal.reset_goal()
            self.highscore=high_s(self.score, self.highscore)
            self.score=0
            self.nb_game+=1
            self.done=True
            return True
        for i in self.obstacle.body:
            for k in self.obstacle.attention:
                if self.score==k:
                    self.mark_obstacle=True
                    if self.player.block.x==i.x and self.player.block.y==i.y:
                        self.player.reroll_player()
                        self.goal.reset_goal()
                        self.highscore = high_s(self.score, self.highscore)
                        self.score = 0
                        self.nb_game+=1
                        self.done=True
                        print("gameover_2")

                else:
                    self.mark_obstacle=False
        return False
    def return_x_y(self):
        """fonction qui retournera la position du joueur...bon, ça aurait pu être utile."""
        return self.player.x,self.player.y, self.goal.x, self.goal.y


def vitesse(score):
    """fonction qui va permettre d'augmenter la vitesse."""
    if score>=30:
        return 0.4
    elif score>=20:
        return 0.6
    elif score>=12:
        return 0.8
    else:
        return 1


def reward (score, rew): #sert à 
    """fonction qui va mettre le reward dans la transition (...éviter les valeurs négatives kofkof)
    si score_{t-1}<score_{t}= rew:1
    si score_{t-1}=>score_{t}; rew:0
    score= le score après l'action (score_{t})
    rew=le score avant l'action (score_{t-1})
    
    """

    if rew<score:
        return 1
    else:
        return 0


def affichage_pygame(display):
    """
    affichage pygame
    diplay= Booléen
    """
    if display:
        #global screen
        screen = pygame.display.set_mode(size=(NB_COL * CELL_SIZE, NB_ROW * CELL_SIZE))
        screen.fill(pygame.Color('white'))  # remplir l'écran principal avec une couleur

        #affichage_score
        image_texte = police.render(str(game.score)+"("+str(game.highscore)+")"+" / nb game: "+str(game.nb_game), 1, (255, 0, 0))
        


        screen.blit(image_texte, (NB_COL, NB_ROW))


        show_grid()  # voir la grille
        game.draw_game_element()

        pygame.display.update()  # permettre la mise à jour des données à chaque pas de temps
        
        pygame.display.flip()
        
        
# -----------------------------------------code
#---pygame


pygame.init()  # initialisation des modules

# taille écran
#10/15
NB_COL = 10
NB_ROW = 15
CELL_SIZE = 40

#affichage/ si True, affiche. sinon false
display=False

if display:
    screen = pygame.display.set_mode(size=(NB_COL * CELL_SIZE, NB_ROW * CELL_SIZE))
    pygame.display.set_caption("save the cat")#titre fenêtre

timer = pygame.time.Clock()

# initialisation jeux
game=game()


#initialisation texte
police=pygame.font.SysFont("",25)


# prendre en compte les touches du joueur
SCREEN_UPDATE = pygame.USEREVENT
pygame.time.set_timer(SCREEN_UPDATE, 100)  # utilisé pour créer un timer de 200millisec.

game_on = True
pygame.time.get_ticks()
#↓---------------------variable DQN
name_action=["RIGHT", "LEFT", "DASH"]
rew=0

mark_vitesse=0#marque qui indiquera la vitesse du jeu

BUFFER_SIZE=50000#taille du buffer
BATCH_SIZE=100#taille du batch (nb éléments qui seront pris dans le buffer pour entrainer le réseau de neurone)
MIN_REPLAY_SIZE=1000#replay minimum avant l'entrainement principal
replay_buffer = deque(maxlen=BUFFER_SIZE)
TARGET_UPDATE_FREQ=1000#fréquence où le 2ème NN va être réinitialisé
eps_start=1#epsilon-greedy début
#eps_end=0.02
eps_end=0.08#epsilon-greedy fin (0.08 et pas moins car je veux encore que mon modèle continue de tester)
GAMMA=0.5#valeur qui va définir l'importance que l'agent va accorder aux récompenses futures par rapport aux récompenses immédiates
eps=1#eps pour l'étape initial
fluctu_remem=0#retenir la vitesse, servira pour l'état
done=False#variable qui permettra de savoir quand une partie se finira

#etape d'origine
#état = position joueur x, pjy, position goal x, pgy, mark obstacle, mark vitesse
st=(game.player.x, game.player.y,game.goal.x,game.goal.y,game.mark_obstacle,mark_vitesse )
nb_gameplus=0
iterboucle_pretrain=0#boucle qui iterera à chaque étape, servira à quitter la boucle quand une valeur X sera atteint//~nombre de transitions.

#2--------------apprentissage DQN
#--2.1: exploration pure
while game_on:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        #score_musique(game.score)
        fluctu=vitesse(game.score)
        fluctu2=100*fluctu
        
        #fluctu2=75
        pygame.time.set_timer(SCREEN_UPDATE,int(fluctu2))  # j'ai dupliqué ici pour penser à le varier pour complexifier le niveau

        #action=[0,1,2] memo actions

        

        if event.type == SCREEN_UPDATE:
            if done:
                st=(game.player.x, game.player.y,game.goal.x,game.goal.y,game.mark_obstacle,mark_vitesse )
                game.done=False

            action=take_action(eps) 
            #game.player.direction = name_action[action]
            game.player.direction = name_action[action]
            game.update()
            done = game.done
            stp1=(game.player.block.x,game.player.block.y, game.goal.block.x,game.goal.block.y,game.mark_obstacle,mark_vitesse)
            transition=(st,action,reward(game.score,rew),done,stp1)
            replay_buffer.append(transition)
            st=stp1
            iterboucle_pretrain+=1
            
        
        rew=game.score
 
                    
            
        if fluctu2>fluctu*100 and fluctu2!=fluctu_remem:
            mark_vitesse+=1
        
            fluctu_remem=fluctu2#modifie flucturemember
                
                
        affichage_pygame(display)
        timer.tick(200)  # "fps"
    
 
    
    if nb_gameplus!=game.nb_game:
        print("nb game:", game.nb_game)
    nb_gameplus=game.nb_game
    if iterboucle_pretrain>=MIN_REPLAY_SIZE:#fin du remplissage initial des transition (1000 normalement)
        print("fin entrainement préléminaire")
        break

#--2.2:entrainement principal

print("entrainement principale qui commence")

#les réseau de neurones= target (celui qui sera modifié tous les x iteration) et network (celui qui fluctuera tout le temps)
DQN_nn_target=Network([len(st),len(name_action)])#target
DQN_nn=Network([len(st),len(name_action)])#online


DQN_nn_target.load_state_dict(DQN_nn.state_dict())#target=online

#l'optimizer est initialisé
optimizer=torch.optim.Adam(DQN_nn.parameters(), lr=5e-4)

#stocker les récompenses pour les stats
rew_buffer = deque([0.0],maxlen=100 )
rew_stock=0
best_epi_reward = 0#stock mean(rew_buffer), servira pour l'affichage dans console

done=False
iter_boucle=0

stock_loss = deque([],maxlen=100 )#stock les fonctions de perte
running_loss=0#servira à faire une moyenne de fonction de perte

#affichage
display=True
while game:
    # pour quitter
    eps=np.linspace(eps_start,eps_end,10000)
    
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        #score_musique(game.score)
        fluctu=vitesse(game.score)
        fluctu2=100*fluctu
        
        #fluctu2=75
        pygame.time.set_timer(SCREEN_UPDATE,int(fluctu2))  # j'ai dupliqué ici pour penser à le varier pour complexifier le niveau

        #action=[0,1,2] memo actions
                
        if event.type == SCREEN_UPDATE:#permet de gérer vitesse de permission d'action/ toutes les processus du DQN sont fait quand il y a une action, l'iter_boucle boucge à chaque action
            if done:
                #st=(3, NB_ROW - 0.2,NB_COL/2,NB_ROW-0.2,game.mark_obstacle,mark_vitesse )
                st=(game.player.x, game.player.y,game.goal.x,game.goal.y,game.mark_obstacle,mark_vitesse )
                rew_buffer.append(rew_stock)#episode
                rew_stock=0
                #print("ola!!")
                game.done=False#
                #episode_reward=0.0
                
            if iter_boucle>=len(eps):
                iter_eps=len(eps)-1#parce que sinnon ça bug...car on est pas sur R
            else:
                iter_eps=iter_boucle
            action=take_action(eps[iter_eps],DQN_nn,st)#action
            game.player.direction = name_action[action]#
            game.update()#mise à jour du jeu (en fonction de l'action)
            done=game.done
            stp1=(game.player.block.x,game.player.block.y, game.goal.block.x,game.goal.block.y,game.mark_obstacle,mark_vitesse)
            transition=(st,action,reward(game.score,rew),done,stp1)#remplissage
            rew_stock+=reward(game.score,rew)
            
            replay_buffer.append(transition)#remplissage replau buffer avec la nouvelle transi
            st=stp1

        #print ("step==iter_boucle? 1",step==iter_boucle1)#false
            iter_boucle+=1
            rew=game.score
            #done = game.game_over()
                    
            
            if fluctu2>fluctu*100 and fluctu2!=fluctu_remem:
                mark_vitesse+=1
            
                fluctu_remem=fluctu2#modifie flucturemember
            
    
            #création du batch
            batch_transition=random.sample(replay_buffer,BATCH_SIZE)#prend des transitions au hasard(Batch transitions (ici 64))
            
            
            #--split des transitions
            obses=[t[0] for t in batch_transition]
            actions=np.asarray([t[1] for t in batch_transition])
            rews=np.asarray([t[2] for t in batch_transition])
            dones=np.asarray([t[3] for t in batch_transition])
            new_obses=np.asarray([t[4] for t in batch_transition])
            
            #--transformation en vecteur torch
            obses_t=torch.as_tensor(obses,dtype=torch.float32)
            actions_t=torch.as_tensor(actions,dtype=torch.int64).unsqueeze(-1)
            rews_t=torch.as_tensor(rews,dtype=torch.float32).unsqueeze(-1)
            dones_t=torch.as_tensor(dones,dtype=torch.float32).unsqueeze(-1)
            new_obses_t=torch.as_tensor(new_obses,dtype=torch.float32)
            
            #--prediction nn (avec target)
            #prediction des qvalues des nouveaux états
            target_q_values=DQN_nn_target(new_obses_t)
            #prends la meilleur qvalue de toutes les nouvelles observations (stp1)
            max_target_q_values=target_q_values.max(dim=1,keepdim=True)[0]#[0] car ce .max renvoie un tuple (X,Y) ou X est la maxvalue et Y l'indice de celle-ci
            
            #formule de bellman, calcul des qvalues de st (obses_t) qui serviront de référence
            targets=rews_t+GAMMA*(1-dones_t)*max_target_q_values
            
            #--comparaison avec les q(st) du model_online+ descent de gradient
            #récupère les q values des observations
            q_values=DQN_nn(obses_t)
            #•prend la qvalue uniquement des actions réellement prises par l'agent
            action_q_values = torch.gather(input=q_values,dim=1,index=actions_t)
            
            loss= nn.functional.smooth_l1_loss(action_q_values,targets)
            
            #descente de gradient= optimisation 
            optimizer.zero_grad()
            loss.backward()#compute le gradient
            optimizer.step()#appliquer les gardient
            running_loss+=loss.item()
            
            if iter_boucle % TARGET_UPDATE_FREQ ==0:
                #update target_net avec les mêmes caractéristiques que online si condition remplie
                DQN_nn_target.load_state_dict(DQN_nn.state_dict())
            
            
            
            if iter_boucle % 100 == 0 :#1000 de base
                print("nombre de jeu",game.nb_game)
                print('Avg rew',np.mean(rew_buffer))
                
                avg_loss=running_loss / 100
                print("fonction de perte :",avg_loss)
                stock_loss.append(avg_loss)
                running_loss=0
                
                

                
            
            if best_epi_reward<np.mean(rew_buffer):
                print("mean reward buffer,:",np.mean(rew_buffer))#il y a plus pertient je crois
                best_epi_reward=np.mean(rew_buffer)
                save_model(DQN_nn_target, "training__DQN_game_prelim_cat")
            #fin if
            
        #affichage
        affichage_pygame(display)
        timer.tick(200)  # "fps"
        

#3-------sauvegarde du modèle entrainé
save_model(DQN_nn_target, "DQN_game_final_cat")
#save_model(model=DQN_nn_target, name="DQN_game_final_cat")

#nb: pour le moment, je coupe moi-même quand les résultats de performance affichés dans la console me plaisent.
#pourquoi? car les stats fluctuent...puisque la difficulté du jeu est progressive. Le nn apprendra bien le début (résultats augm.), puis découvre la suite(chute résultat), et l'apprends (augm), etc.

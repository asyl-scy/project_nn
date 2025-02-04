"""
date = 04/02/2025

@author: asyl_scy (Kara)

topic= On va apprendre à un réseau de neurone à jouer à mon jeu!!

Ce code servira à tester mon nn! On va enfin voir du bon niveau

"""

import pygame
from pygame.locals import*
import sys
import random
from random import randint
from torch import nn
import torch




"""
On va tenter de faire du DQN avec le jeux cube game (save the cat)
action=["droite", "gauche", "dash"] = [0,1,2]

problème= nb images par secondes et implentation des transitions

idée=average score (plus pertinent pour savoir à quel point l'apprentissage avance')
"""

#tiempo

#rect(positionx,positiony, taille, taille)



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
            print("score : ",self.score)
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

display=True
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

stock=[]
mark_vitesse=0


fluctu_remem=0#retenir la vitesse
done=False



#etape d'origine
#état = position joueur x, pj y, position goal x, pg y, mark obstacle, mark vitesse
st=(game.player.x, game.player.y,game.goal.x,game.goal.y,game.mark_obstacle,mark_vitesse )

#nn

DQN_nn=Network([len(st),len(name_action)])
DQN_nn.load_state_dict(torch.load("./stock_poids/tent4/training_DQN_game_prelim_cat_4_asyl.pth", weights_only=True))
nb_gameplus=0
iterboucle_pretrain=0
#1----remplissage premières transitions= pas d'apprentissage
while game_on:
#for _ in range(100):
    # pour quitter
    current_time = pygame.time.get_ticks()
    for event in pygame.event.get():#les actions et tout le tralala dans la boucle?!
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
                #remplissage état à t_{0}
                st=(game.player.x, game.player.y,game.goal.x,game.goal.y,game.mark_obstacle,mark_vitesse )
                game.done=False

            #choix direction par nn
            action=DQN_nn.act(st)
            #introduction du choix de l'action dans le jeu
            game.player.direction = name_action[action]
            game.update()
            done = game.done#récupération de done (est-ce qu'il a perdu?)
            #remplissage état t_{+1}
            stp1=(game.player.block.x,game.player.block.y, game.goal.block.x,game.goal.block.y,game.mark_obstacle,mark_vitesse)
            
            st=stp1#état t_{+1}=état t_{0}
            #iterboucle_pretrain+=1
            
 
                    
            
        if fluctu2>fluctu*100 and fluctu2!=fluctu_remem:#pas utilisé je crois
            mark_vitesse+=1
        
            fluctu_remem=fluctu2#modifie flucturemember
                
                
        #affichage si display=True
        affichage_pygame(display)
        timer.tick(200)  # "fps"
    
    if nb_gameplus!=game.nb_game:
        print("nb game:", game.nb_game)
    nb_gameplus=game.nb_game



# Simulation herbivore/carnivore : projet evolusurv
Voici le projet de l'année je pense. J'ai souhaité ici faire une simulation de la relation entre herbivores et carnivores. In-fine, le but serait de faire une comparaison entre ce modèle et un autre plus simple (EDO(Equation différentielle Ordinaire)) pour voir les différences, etc. (J'aurais dû commencer par l'EDO, plus simpel aha!)

-> les carnivores peuvent manger les herbivores, les herbivores peuvent manger les plantes. 
->Les herbivores et les carnivores peuvent bouger, se reproduire, et "attaquer"
-> les plantes ne peuvent que se reproduire (clonage uniquement)


**evolvsurv_v1.py**= première version de ma simulation. Il manque des choses comme l'efficacité de reproduction en fonction de l'age ainsi que la mort par la vieillesse. De plus, le temps n'est pas bien initialisé.
Mais on peut voir des choses intéressantes. Ce code n'est pas encore parfait, je pense que c'est en partie à cause de la taille de l'environnement ainsi que du nombre d'individu (j'ai pas envie que mon ordi crame, on ne peut pas aller plus loin).

-> aller plus loin= faire une compétition entre 4-5 DQN; améliorer les analyses de l'efficacité des DQN; mieux gérer l'age et le temps;

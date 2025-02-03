# DQN_game


**Train_DQN_model.py** = Code qui va entrainer le nn a jouer à mon jeu (apprentissage en mode DQN).
  *  Première phase = exploration (<1H)
  *  deuxième phase = exploration + exploitation (descente de Epsilon-Greedy). (ça dépend. Moi, j'ai pris 6H perso...mais on peut raisonnablement baisser ce temps)
  *  Troisième phase = sauvegarde. (...1 seconde)
Les temps dépendent des capacités du terminal, mais aussi de la taille du Buffer, et du Batch.

**Test_DQN_model.py** = code qui va tester le nn. 
De mon côté c'est très simple = je n'ai pas pu voir le nn perdre.
Son meilleur score est corrélé avec ma patience, ni plus ni moins.



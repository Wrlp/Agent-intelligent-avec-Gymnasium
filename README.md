# Description
On crée un environnement de jeu (ici TicTacToe), puis on entraînera une IA dessus avec du reinforcement learning

# Env
Je m’occupe de créer l’environnement compatible RL (règles du jeu, états, actions, rewards). Les autres feront l’agent et l’entraînement.
Pour l’instant j’utilise des actions aléatoires, mais l’objectif est d’entraîner un agent (type Q-learning ou Deep RL)

state → plateau 3×3
action → entier de 0 à 8
reward :
    +1 victoire
    0 match nul
    -1 coup invalide
    done → fin de partie

J’ai implémenté un environnement Tic Tac Toe compatible avec Gymnasium.
L’état est une grille 3×3, les actions sont les 9 cases.
La fonction step applique un coup, calcule la récompense et vérifie la fin de partie.
Cet environnement pourra être utilisé directement par MCTS et DQN.

Reste à faire : 
- Corriger les coups invalides définitevement
- Respecter parfaitement le format de Gymnasium
- Gérer le joueur adverse
  - J1 : IA joue
  - J2 : aléatoire
- Ajouter un mode entrainement
  - reset rapide
  - pas de print -> render desactivé pour que l'agent puisse faire des milliers de parties
- Documentation
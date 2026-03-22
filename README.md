# Environnement Othello — Gymnasium

## Description
Environnement Othello compatible Gymnasium, conçu pour être utilisé
par **MCTS** (logique numpy) et **DQN** (pixels Atari).

## Structure
```
Agent-intelligent-avec-Gymnasium/
├── envs/
│   └── game_env.py     <- Environnement principal
    └── tests_env.py <- Tests & démo
```

## Deux modes
| Mode      | Observation           | Usage          |
|-----------|-----------------------|----------------|
| `logic`   | Plateau numpy 8×8     | MCTS, tests    |
| `atari`   | Pixels RGB (ALE)      | DQN            |

## État / Actions / Rewards
```
state   -> plateau 8×8  (0=vide, 1=Noir, 2=Blanc)
action  -> entier 0-63 (case) ou 64 (passer son tour)
reward :
  +1    victoire
  -1    défaite
   0    match nul / partie en cours
  -0.5  coup invalide (non terminal)
```

## Installation
```bash
pip install gymnasium[atari] ale-py
ale-import-roms roms/   # si nécessaire
```

## Utilisation rapide
```python
from game_env import OthelloEnv

# Mode logique (MCTS)
env = OthelloEnv(mode="logic")
obs, _ = env.reset()
legal = env.get_legal_actions()  # liste d'entiers
obs, reward, done, _, info = env.step(legal[0])

# Clone pour MCTS (exploration sans modifier l'état)
clone = env.clone()

# Charger un état externe (MCTS)
env.set_state(board_array, current_player)

# Mode Atari (DQN)
env_atari = OthelloEnv(mode="atari", render_mode="rgb_array")
pixels, _ = env_atari.reset()
```

## Lancer les tests
```bash
python tests_env.py
```

## Interface pour les coéquipiers

### Pour MCTS (Personnes 2 & 3)
- `env.get_legal_actions()` → liste des actions légales
- `env.clone()` → copie indépendante pour simuler une branche
- `env.set_state(board, player)` → charger un état
- `env.current_player` → joueur courant (1 ou 2)
- `env._final_reward()` → reward de fin de partie

### Pour DQN (Personne 4)
- Utiliser `mode="atari"` pour avoir les pixels
- `env.observation_space` et `env.action_space` compatibles Gymnasium
- `env.get_legal_actions()` disponible pour masquer les actions illégales

## Différences vs TicTacToe
| Aspect           | TicTacToe       | Othello                        |
|------------------|-----------------|--------------------------------|
| Plateau          | 3×3             | 8×8                            |
| Actions          | 9               | 64 + 1 (passer)                |
| Coup valide      | Case vide       | Retourne ≥1 pièce adverse      |
| Passer           | Non             | Oui (si aucun coup possible)   |
| Fin de partie    | 3 en ligne / full | Plus aucun coup pour personne |





env.reset() → position initiale
env.get_legal_actions() → 4 coups au départ
env.clone() → interface prête pour MCTS
Les 8 tests qui passent

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
- Changer de jeu -> Othello
# 8INF974 - Atelier pratique en IA II - Hiver 2026 - 01
## Projet 2
> Anna-Eve Mercier & Flavien Baron & Ewan Schwaller & Laure Warlop
> 23/03/26
> Professeur : Kévin Bouchard

## Introduction
L'objectif est d'explorer l'apprentissage par renforcement à travers l'implémentation de deux approches distinctes : le Monte Carlo Tree Search (MCTS) et le Deep Q-Learning (DQN), appliquées à un jeu de plateau Atari via la librairie Gymnasium.
Nous avons choisi Othello comme environnement principal. Ce jeu à somme nulle déterministe représente un bon équilibre entre complexité et faisabilité — plus riche que TicTacToe, mais plus accessible que Video Chess. Il se prête bien aux deux méthodes : MCTS exploite la logique du jeu directement, tandis que DQN apprend à partir des pixels via l'émulateur Atari.
Le travail a été divisé en quatre parties complémentaires : l'environnement de jeu, la structure de l'arbre MCTS, la simulation et backpropagation MCTS, et l'agent DQN. Chaque partie s'appuie sur les autres — l'environnement servant de base commune à toutes les approches.

# Environnement Othello — Gymnasium
# Description de l'environnement
J'ai d'abord commencer par réaliser un TicTacToe avant de partir sur Othello.
J'ai implémenté un environnement Othello compatible Gymnasium en deux modes : logic (plateau numpy 8×8) pour MCTS et atari (pixels RGB via ALE/Othello-v5) pour DQN. Les deux modes partagent la même logique de jeu.
L'environnement gère :

- Les coups valides (retournement d'au moins une pièce adverse dans 8 directions)
- Le passage de tour automatique si un joueur n'a aucun coup
- La fin de partie quand les deux joueurs passent consécutivement
- Un système de récompenses : +1 victoire, -1 défaite, 0 nul, -0.5 coup invalide

Deux méthodes clés ont été ajoutées pour l'intégration avec MCTS : clone() pour explorer des branches sans modifier l'état principal, et set_state(board, player) pour charger un état externe.

## Structure de l'environnement
```
Agent-intelligent-avec-Gymnasium/
├── envs/
│   └── game_env.py     <- Environnement principal
    └── tests_env.py <- Tests & démo
```

## Utilité des deux modes 
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

## Installation de l'environnement
```bash
pip install gymnasium[atari] ale-py
ale-import-roms roms/   # si nécessaire
```
## Ce qui a réussi
- Passage du TicTacToe à Othello
- Les 8 tests unitaires (reset, coups légaux, retournement, clone, partie complète, etc.)
- Intégration avec MCTS et DQN fonctionnelle
- ale-py et les ROMs Atari installés et fonctionnels

## Bugs / Défis / Apprentissages
- Bug principal : Les coups légaux étaient calculés avec v==0 (cases vides) au lieu de `get_legal_actions()`. A Othello, une case vide n'est pas forcément un coup légal car il faut forcément retourner au moins une pièce. Le bug venait du passage entre TicTacToe et Othello.
- `set_state()` prenait deux arguments obligatoire (board, player) mais le code MCTS appelait `set_state(state)` avec un seul argument via `deepcopy`. Le problème a été résolu en rendant `player` optionnel avec `player=None`.
- Ce que j'aurais pu faire différemment c'est de commencer directement par créer un Othello plutôt que de passer par un TicTacToe avant pour éviter les différents problèmes liés au changement de jeu. 

## Différences vs TicTacToe
| Aspect           | TicTacToe       | Othello                        |
|------------------|-----------------|--------------------------------|
| Plateau          | 3×3             | 8×8                            |
| Actions          | 9               | 64 + 1 (passer)                |
| Coup valide      | Case vide       | Retourne ≥1 pièce adverse      |
| Passer           | Non             | Oui (si aucun coup possible)   |
| Fin de partie    | 3 en ligne / full | Plus aucun coup pour personne |

## Code pertinent 
```Python
# calcul des coups valides
def _would_flip(self, row, col, player, opponent):
    for dr, dc in DIRECTIONS:
        r, c = row + dr, col + dc
        found_opponent = False
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            if self.board[r][c] == opponent:
                found_opponent = True
            elif self.board[r][c] == player and found_opponent:
                return True
            else:
                break
            r += dr
            c += dc
    return False

# clone pour MCTS
def clone(self):
    clone = OthelloEnv(mode="logic")
    clone.board = self.board.copy()
    clone.current_player = self.current_player
    clone.pass_count = self.pass_count
    return clone
```
## Captures d'écrans
![tests_all](test-env-1.png)
![8-tests-1](8-tests-1.png)
![8-tests-2](8-tests-2.png)

## Interface pour les coéquipiers

### Pour MCTS 
- `env.get_legal_actions()` → liste des actions légales
- `env.clone()` → copie indépendante pour simuler une branche
- `env.set_state(board, player)` → charger un état
- `env.current_player` → joueur courant (1 ou 2)
- `env._final_reward()` → reward de fin de partie

### Pour DQN 
- Utiliser `mode="atari"` pour avoir les pixels
- `env.observation_space` et `env.action_space` compatibles Gymnasium
- `env.get_legal_actions()` disponible pour masquer les actions illégales


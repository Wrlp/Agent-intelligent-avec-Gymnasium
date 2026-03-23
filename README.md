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


# Fonctionnement du Monte Carlo Tree Search (MCTS)

## Vue d'ensemble

Le Monte Carlo Tree Search (MCTS) est un algorithme de recherche basé sur l'exploration aléatoire. Contrairement aux algorithmes exhaustifs (comme minimax), MCTS explore l'arbre de jeu de manière probabiliste pour trouver les meilleurs coups sans évaluer tous les états possibles.

C'est utile pour les jeux qui ont beaucoup d'actions possibles par position (Othello a 65 actions possibles), où il n'est pas réaliste d'explorer tous les coups.

## Comment ça marche

Le MCTS fonctionne en 4 phases qui se répètent plusieurs fois :

### Phase 1 : Sélection

On commence à la racine et on descend l'arbre en choisissant le meilleur enfant à chaque étape. Le meilleur enfant est choisi selon la formule UCB1 (Upper Confidence Bound). On continue à descendre jusqu'à trouver un nœud qu'on n'a pas complètement exploré.

La formule UCB1 est :
$$\text{UCB1} = \frac{\text{valeur}}{\text{visites}} + C \cdot \sqrt{\frac{\ln(\text{visites\_parent})}{\text{visites\_enfant}}}$$

- La première partie (valeur/visites) c'est la qualité moyenne du nœud qu'on a déjà mesurée
- La deuxième partie (le √) encourage d'explorer les nœuds qu'on a moins visités
- C est un paramètre de balance (normalement 1.41)

Voir [mcts/tree.py](mcts/tree.py#L46) pour la méthode `best_child()`

### Phase 2 : Expansion

Si le nœud sélectionné n'est pas un état terminal et qu'il y a des actions qu'on n'a pas encore essayées, on en choisit une et on crée un nouveau nœud enfant.

Voir [mcts/tree.py](mcts/tree.py#L81) pour la méthode `add_child()`

### Phase 3 : Simulation (Rollout)

À partir du nouveau nœud, on joue une partie complète en choisissant les coups au hasard jusqu'à la fin du jeu. On récupère la récompense finale.

[mcts/mcts_agent.py](mcts/mcts_agent.py#L8) - fonction `rollout()`

```python
def rollout(env, state):
    sim_env = deepcopy(env)
    sim_env.set_state(state, env.current_player)
    
    done = False
    total_reward = 0
    while not done:
        actions = sim_env.get_legal_actions()
        action = random.choice(actions)
        obs, reward, terminated, truncated, _ = sim_env.step(action)
        done = terminated or truncated
        total_reward += reward
    
    return total_reward
```

### Phase 4 : Backpropagation

On remonte l'arbre depuis le nœud feuille jusqu'à la racine. Pour chaque nœud traversé, on met à jour ses statistiques :
- On ajoute 1 au compteur de visites
- On ajoute la récompense à la valeur totale

[mcts/mcts_agent.py](mcts/mcts_agent.py#L37) - fonction `backpropagate()`

```python
def backpropagate(node, reward):
    current = node
    while current is not None:
        current.visits += 1
        current.value += reward
        current = current.parent
```

## Les nœuds de l'arbre

Chaque nœud contient :
- `state` : l'état du jeu à ce point
- `visits` : nombre de fois qu'on a visité ce nœud
- `value` : somme totale des récompenses accumulées
- `children` : liste des nœuds enfants qu'on a déjà explorés
- `untried_actions` : les actions qu'on n'a pas encore essayées
- `parent` : le nœud parent (pour la backpropagation)

Après avoir fait plein d'itérations, le meilleur coup est celui qui mène au nœud enfant le plus visité :

```python
best_action = root.most_visited_child()
```

## Exemple simplifié

```
État initial
├─ Action 1 → Nœud A (visits=50) <- Nœud le plus visité
├─ Action 2 → Nœud B (visits=30)
├─ Action 3 → Nœud C (visits=20)
└─ Action 4 → Pas encore exploré

On choisira Action 1 parce que le Nœud A a été visité le plus
```

## Paramètres importants

| Paramètre | Valeur | Explication |
|-----------|--------|---|
| Nombre d'itérations | N | Plus on fait d'itérations, meilleur le coup, mais ça prend plus de temps |
| Exploration (C) | 1.41 | Plus grand = plus d'exploration, plus petit = plus d'exploitation |
| Profondeur simulation | Sans limite | Tant qu'on n'a pas la fin de partie |

## Avantages et inconvénients

Avantages :
- Fonctionne bien pour les jeux avec beaucoup d'actions possibles
- Pas besoin d'une fonction pour évaluer les positions
- On peut arrêter quand on veut et avoir une bonne réponse

Inconvénients :
- Les simulations aléatoires prennent du temps si la partie est longue
- Pour les jeux avec beaucoup de hasard, il faut faire beaucoup plus d'itérations
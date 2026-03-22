import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

# Directions possibles pour retourner des pièces (8 directions)
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              ( 0, -1),          ( 0, 1),
              ( 1, -1), ( 1, 0), ( 1, 1)]

BOARD_SIZE = 8

class OthelloEnv(gym.Env):
    """
    Environnement Othello compatible Gymnasium.

    Deux modes disponibles :
      - mode 'logic'  : état = plateau numpy 8x8 (pour MCTS et tests rapides)
      - mode 'atari'  : wraps ALE/Othello-v5, état = pixels RGB (pour DQN)

    Dans les deux modes, la logique du jeu (coups valides, retournement des
    pièces, détection de fin de partie) est gérée en interne via numpy.

    Joueurs :
      1  -> Noir  (joue en premier)
      2  -> Blanc
      0  -> Case vide

    Récompenses (du point de vue du joueur courant) :
      +1  victoire
      -1  défaite
       0  match nul ou partie en cours
      -0.5 coup invalide (non terminal, mais pénalisé)
    """

    metadata = {"render_modes": ["human", "rgb_array", "ansi"]}

    # construction
    def __init__(self, mode: str = "logic", render_mode: str = None):
        """
        Args:
            mode        : 'logic' (numpy) ou 'atari' (pixels ALE)
            render_mode : 'human', 'rgb_array' ou 'ansi'
        """
        super().__init__()

        assert mode in ("logic", "atari"), "mode doit être 'logic' ou 'atari'"
        self.mode = mode
        self.render_mode = render_mode

        # Plateau interne (toujours maintenu, peu importe le mode)
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = 1   # 1 = Noir, 2 = Blanc
        self.pass_count = 0       # Nombre de passes consécutives (fin de partie si 2)

        # Espaces action / observation 
        # 64 cases + 1 action "passer son tour"
        self.action_space = Discrete(BOARD_SIZE * BOARD_SIZE + 1)
        self.PASS_ACTION = BOARD_SIZE * BOARD_SIZE  # index 64

        if mode == "logic":
            # Observation : plateau 8×8, valeurs 0/1/2
            self.observation_space = Box(
                low=0, high=2,
                shape=(BOARD_SIZE, BOARD_SIZE),
                dtype=np.float32
            )
            self._atari_env = None

        else:  # mode atari
            # On charge l'env Atari ; on récupère son espace d'observation (pixels)
            self._atari_env = gym.make(
                "ALE/Othello-v5",
                render_mode=render_mode,
                obs_type="rgb",          # pixels RGB
                frameskip=1,
                repeat_action_probability=0.0,  # déterministe
            )
            self.observation_space = self._atari_env.observation_space

    # reset
    def reset(self, seed=None, options=None):
        """Réinitialise le plateau dans la position de départ standard."""
        super().reset(seed=seed)

        self.board.fill(0)
        self.current_player = 1
        self.pass_count = 0

        # Position initiale Othello : 4 pièces au centre
        mid = BOARD_SIZE // 2
        self.board[mid - 1][mid - 1] = 2  # Blanc
        self.board[mid - 1][mid]     = 1  # Noir
        self.board[mid][mid - 1]     = 1  # Noir
        self.board[mid][mid]         = 2  # Blanc

        if self.mode == "atari" and self._atari_env is not None:
            atari_obs, info = self._atari_env.reset(seed=seed)
            return atari_obs, info

        return self.get_obs(), {}

    # step
    def step(self, action: int):
        """
        Applique l'action au plateau.
        Args:
            action : entier 0-63 (case) ou 64 (passer)

        Returns:
            observation, reward, terminated, truncated, info
        """
        legal = self.get_legal_actions()

        # Coup invalide
        if action not in legal:
            obs = self.get_obs()
            info = {"illegal_move": True, "current_player": self.current_player}
            # On ne termine pas la partie, mais on pénalise
            return obs, -0.5, False, False, info

        # Passer son tour
        if action == self.PASS_ACTION:
            self.pass_count += 1
            # Si les deux joueurs passent de suite -> fin de partie
            if self.pass_count >= 2:
                reward = self._final_reward()
                return self.get_obs(), reward, True, False, {"passed": True}
            self.current_player = 3 - self.current_player
            return self.get_obs(), 0, False, False, {"passed": True}

        # Coup normal
        self.pass_count = 0
        row, col = divmod(action, BOARD_SIZE)
        self._apply_move(row, col, self.current_player)

        # Vérification fin de partie
        terminated = False
        reward = 0

        next_player = 3 - self.current_player
        next_legal = self._compute_legal_actions(next_player)

        if not next_legal:
            # Le joueur suivant n'a aucun coup ; vérifie si le joueur actuel peut jouer
            current_legal = self._compute_legal_actions(self.current_player)
            if not current_legal:
                # Plus aucun coup possible pour personne -> fin
                terminated = True
                reward = self._final_reward()
            else:
                # Le joueur suivant doit passer
                self.pass_count += 1
                # On ne change PAS de joueur (le courant rejoue)
        else:
            self.current_player = next_player

        # Sync Atari (mode atari)
        atari_obs = None
        if self.mode == "atari" and self._atari_env is not None:
            # On applique l'action dans l'env Atari pour récupérer les pixels
            # Note : l'env Atari gère son propre état interne ; on le synchronise
            # en appliquant la même action (les deux états peuvent diverger
            # légèrement à cause des frames Atari — acceptable pour le rendu)
            atari_obs, _, _, _, atari_info = self._atari_env.step(action)

        obs = atari_obs if (self.mode == "atari" and atari_obs is not None) else self.get_obs()
        info = {
            "current_player": self.current_player,
            "board": self.board.copy(),
            "score": self._score(),
        }
        return obs, reward, terminated, False, info

    # observation
    def get_obs(self):
        """Retourne le plateau numpy (mode logic) ou les pixels (mode atari)."""
        return self.board.copy().astype(np.float32)

    # logique du jeu
    def get_legal_actions(self) -> list[int]:
        """
        Retourne la liste des actions légales pour le joueur courant.
        Inclut l'action PASS (64) si aucun coup normal n'est possible.
        """
        moves = self._compute_legal_actions(self.current_player)
        if not moves:
            return [self.PASS_ACTION]
        return moves

    def _compute_legal_actions(self, player: int) -> list[int]:
        """Calcule les coups valides pour 'player' sans modifier l'état."""
        opponent = 3 - player
        legal = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.board[row][col] != 0:
                    continue
                if self._would_flip(row, col, player, opponent):
                    legal.append(row * BOARD_SIZE + col)
        return legal

    def _would_flip(self, row: int, col: int, player: int, opponent: int) -> bool:
        """Vérifie si poser une pièce en (row, col) retournerait au moins une pièce."""
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

    def _apply_move(self, row: int, col: int, player: int):
        """Place la pièce et retourne toutes les pièces capturées."""
        opponent = 3 - player
        self.board[row][col] = player

        for dr, dc in DIRECTIONS:
            to_flip = []
            r, c = row + dr, col + dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                if self.board[r][c] == opponent:
                    to_flip.append((r, c))
                elif self.board[r][c] == player:
                    # On retourne toutes les pièces capturées dans cette direction
                    for fr, fc in to_flip:
                        self.board[fr][fc] = player
                    break
                else:
                    break
                r += dr
                c += dc

    def _score(self) -> dict:
        """Retourne le score {1: nb_noir, 2: nb_blanc}."""
        return {
            1: int(np.sum(self.board == 1)),
            2: int(np.sum(self.board == 2)),
        }

    def _final_reward(self) -> float:
        """
        Calcule la récompense finale du point de vue du joueur courant.
        +1 victoire, -1 défaite, 0 nul.
        """
        score = self._score()
        s1, s2 = score[1], score[2]
        winner = 1 if s1 > s2 else (2 if s2 > s1 else 0)
        if winner == 0:
            return 0.0
        return 1.0 if winner == self.current_player else -1.0

    # etat (pour MCTS -> clonage rapide)
    def set_state(self, board: np.ndarray, player: int = None):
        """
        Charge un état extérieur dans l'environnement.
        Utilisé par MCTS pour simuler des branches sans créer de nouveaux env.

        Args:
            board  : numpy array 8x8
            player : joueur courant (1 ou 2)
        """
        self.board = np.array(board, dtype=np.int8).reshape(BOARD_SIZE, BOARD_SIZE)
        self.current_player = player if player is not None else self.current_player
        self.pass_count = 0

    def clone(self):
        """
        Retourne un clone léger de l'environnement (mode logic uniquement).
        Utilisé par MCTS pour explorer sans modifier l'état principal.
        """
        clone = OthelloEnv(mode="logic", render_mode=None)
        clone.board = self.board.copy()
        clone.current_player = self.current_player
        clone.pass_count = self.pass_count
        return clone

    # render
    def render(self):
        """Affiche le plateau en mode texte (ansi/human)."""
        if self.mode == "atari" and self._atari_env is not None:
            return self._atari_env.render()

        symbols = {0: ".", 1: "X", 2: "O"}
        print("  " + " ".join(str(i) for i in range(BOARD_SIZE)))
        for i, row in enumerate(self.board):
            print(f"{i} " + " ".join(symbols[v] for v in row))

        score = self._score()
        player_str = "Noir (X)" if self.current_player == 1 else "Blanc (O)"
        print(f"Score -> Noir: {score[1]}  Blanc: {score[2]}")
        print(f"Tour  -> {player_str}")
        print()

    # cleanup
    def close(self):
        if self._atari_env is not None:
            self._atari_env.close()


# import numpy as np
# import gymnasium as gym
# from gymnasium.spaces import Discrete, Box

# class TicTacToeEnv(gym.Env):
#     """Environnement simple de Morpion (Tic-Tac-Toe) compatible Gymnasium."""
#     def __init__(self, render_mode=None):
#         super(TicTacToeEnv, self).__init__()
#         self.board = np.zeros((3,3), dtype=int) 
#         self.current_player = 1 # 1: X, 2: O
#         self.action_space = Discrete(9)
#         # 0: Vide, 1: Joueur 1 (X), 2: Joueur 2 (O)
#         self.observation_space = Box(low=0, high=2, shape=(3,3), dtype=int)
#         self.render_mode = render_mode

#     def reset(self, seed=None, options=None):
#         """Réinitialise l'environnement."""
#         super().reset(seed=seed)
#         self.board.fill(0)
#         self.current_player = 1
#         return self.get_obs(), {}

#     def get_obs(self):
#         """Retourne l'observation actuelle (du point de vue du joueur courant)."""
#         # Pour simplifier le DQN, on retourne le plateau tel quel
#         return self.board.copy().astype(np.float32)

#     def step(self, action):
#         """Applique l'action."""
#         row = action // 3
#         col = action % 3

#         # Coup invalide
#         if self.board[row, col] != 0:
#             return self.board.copy(), -1, False, {}  # coup invalide, game over pour le test

#         # Applique le coup
#         self.board[row, col] = self.current_player

#         # Vérification victoire
#         terminated = False
#         reward = 0
        
#         if self._check_winner(self.current_player):
#             reward = 1
#             terminated = True
#         elif np.all(self.board != 0):
#             reward = 0 # Match nul
#             terminated = True
        
#         # Change de joueur (si non terminé)
#         if not terminated:
#             self.current_player = 3 - self.current_player
            
#         return self.get_obs(), reward, terminated, False, {}

#     def render(self):
#         """Affiche le plateau."""
#         symbols = {0: ".", 1: "X", 2: "O"}
#         for row in self.board:
#             print(" ".join(symbols[val] for val in row))
#         print()

#     def _check_winner(self, player):
#         for i in range(3):
#             if np.all(self.board[i,:] == player) or np.all(self.board[:,i] == player):
#                 return True
#         if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
#             return True
#         return False
    
#     def set_state(self, state):
#         self.board = np.array(state).reshape(3, 3).copy()

#     def get_legal_actions(self):
#         """
#         Retourne les actions possibles (cases vides).
#         """
#         actions = []
#         for i in range(9):
#             row = i // 3
#             col = i % 3
#             if self.board[row, col] == 0:
#                 actions.append(i)
#         return actions

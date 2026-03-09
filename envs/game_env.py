import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

class TicTacToeEnv(gym.Env):
    """Environnement simple de Morpion (Tic-Tac-Toe) compatible Gymnasium."""
    def __init__(self, render_mode=None):
        super(TicTacToeEnv, self).__init__()
        self.board = np.zeros((3,3), dtype=int) 
        self.current_player = 1 # 1: X, 2: O
        self.action_space = Discrete(9)
        # 0: Vide, 1: Joueur 1 (X), 2: Joueur 2 (O)
        self.observation_space = Box(low=0, high=2, shape=(3,3), dtype=int)
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        """Réinitialise l'environnement."""
        super().reset(seed=seed)
        self.board.fill(0)
        self.current_player = 1
        return self.get_obs(), {}

    def get_obs(self):
        """Retourne l'observation actuelle (du point de vue du joueur courant)."""
        # Pour simplifier le DQN, on retourne le plateau tel quel
        return self.board.copy().astype(np.float32)

    def step(self, action):
        """Applique l'action."""
        row = action // 3
        col = action % 3

        # Coup invalide
        if self.board[row, col] != 0:
            return self.board.copy(), -1, False, {}  # coup invalide, game over pour le test

        # Applique le coup
        self.board[row, col] = self.current_player

        # Vérification victoire
        terminated = False
        reward = 0
        
        if self._check_winner(self.current_player):
            reward = 1
            terminated = True
        elif np.all(self.board != 0):
            reward = 0 # Match nul
            terminated = True
        
        # Change de joueur (si non terminé)
        if not terminated:
            self.current_player = 3 - self.current_player
            
        return self.get_obs(), reward, terminated, False, {}

    def render(self):
        """Affiche le plateau."""
        symbols = {0: ".", 1: "X", 2: "O"}
        for row in self.board:
            print(" ".join(symbols[val] for val in row))
        print()

    def _check_winner(self, player):
        for i in range(3):
            if np.all(self.board[i,:] == player) or np.all(self.board[:,i] == player):
                return True
        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

# classe qui wrap Gymnasium pour le jeu choisi
# tictactoe
import numpy as np
from gymnasium.spaces import Discrete, Box

class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros((3,3), dtype=int) # plateau 3x3 vide
        self.current_player = 1
        self.action_space = Discrete(9) # 9 cases
        self.observation_space = Box(low=0, high=2, shape=(3,3), dtype=int)

    def reset(self):
        """Remet le plateau à zéro et retourne l'état initial"""
        self.board[:] = 0
        self.current_player = 1
        return self.board.copy(), {}

    def step(self, action):
        """Applique un coup et retourne (state, reward, done, info)"""
        row = action // 3
        col = action % 3

        # Vérifie que la case est vide
        if self.board[row, col] != 0:
            return self.board.copy(), -1, True, {}  # coup invalide, game over pour le test

        # Applique le coup
        self.board[row, col] = self.current_player

        # Vérifie fin de partie
        done = False
        reward = 0
        if self._check_winner(self.current_player):
            reward = 1
            done = True
        elif np.all(self.board != 0):
            done = True  # match nul

        # Change de joueur
        self.current_player = 3 - self.current_player

        return self.board.copy(), reward, done, {}

    def render(self):
        """Affiche le plateau pour debug / démonstration"""
        symbols = {0: ".", 1: "X", 2: "O"}
        for row in self.board:
            print(" ".join(symbols[val] for val in row))
        print()  # ligne vide pour lisibilité

    def _check_winner(self, player):
        """Vérifie si le joueur courant a gagné"""
        # Lignes et colonnes
        for i in range(3):
            if np.all(self.board[i,:] == player) or np.all(self.board[:,i] == player):
                return True
        # Diagonales
        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False
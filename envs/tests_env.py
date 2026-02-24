# pour test
from game_env import TicTacToeEnv
import numpy as np

env = TicTacToeEnv()

state, info = env.reset()
env.render()

action = 0  # exemple : joueur 1 joue case 0
state, reward, done, info = env.step(action)
env.render()
print("Reward:", reward, "Done:", done)

done = False

while not done:
    valid_actions = np.where(env.board.flatten() == 0)[0]
    action = np.random.choice(valid_actions)
    state, reward, done, info = env.step(action)
    env.render()

print("Fin de partie | Reward:", reward)
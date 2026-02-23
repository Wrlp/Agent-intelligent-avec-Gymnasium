# pour test
from game_env import TicTacToeEnv

env = TicTacToeEnv()

state, info = env.reset()
env.render()

action = 0  # exemple : joueur 1 joue case 0
state, reward, done, info = env.step(action)
env.render()
print("Reward:", reward, "Done:", done)
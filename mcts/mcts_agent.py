# selection, expansion, simulation, backpropagation (personne 3)

import math
import random
from copy import deepcopy
from mcts.tree import Node

def rollout(env, state):
    """
    Simulation aléatoire complète depuis un état donné.
    Retourne la récompense finale.
    """
    sim_env = deepcopy(env)
    sim_env.set_state(state)

    done = False
    total_reward = 0

    while not done:
        action = random.choice(sim_env.get_legal_actions())
        obs, reward, terminated, truncated, _ = sim_env.step(action)
        done = terminated or truncated
        total_reward += reward

    return total_reward


def backpropagate(node, reward):
    """
    Remonte le résultat de la simulation dans l'arbre.
    """
    current = node
    while current is not None:
        current.update(reward)
        current = current.parent



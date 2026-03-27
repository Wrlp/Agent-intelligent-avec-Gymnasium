# selection, expansion, simulation, backpropagation (personne 3)

import math
import random
from copy import deepcopy
from mcts.tree import Node

def heuristic_action(env, legal_actions):
    """
    Heuristic
    - Prioritize corners
    - Otherwise, random selection
    """
    corners = [0, 7, 56, 63]

    for action in legal_actions:
        if action in corners:
            return action

    return random.choice(legal_actions)

def rollout(env, state):
    """
    Full random simulation starting from a given state.
    Returns the final reward.
    """
    sim_env = env.clone()
    sim_env.set_state(state)

    done = False
    total_reward = 0

    while not done:
        legal_actions = sim_env.get_legal_actions()
        if not legal_actions:
            break

        action = heuristic_action(sim_env, legal_actions)
        obs, reward, terminated, truncated, _ = sim_env.step(action)
        done = terminated or truncated
        total_reward += reward

    return total_reward


def backpropagate(node, reward):
    """
    Backpropagate the result of the simulation up the tree.
    """
    current = node
    while current is not None:
        current.update(reward)
        current = current.parent



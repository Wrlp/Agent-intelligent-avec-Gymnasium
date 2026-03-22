import numpy as np

# ENVIRONNEMENT
# from envs.game_env import TicTacToeEnv
from envs.game_env import OthelloEnv

# DQN
from dqn.train import train as train_dqn
from dqn.agent import DQNAgent

# MCTS
from mcts.tree import MCTSTree
from mcts.mcts_agent import rollout, backpropagate

def run_random_game():
    env = OthelloEnv(mode="logic")
    # env = OthelloEnv(mode="atari", render_mode="rgb_array")
    obs, _ = env.reset()
    print(obs.shape)
    done = False

    print("\n Partie aléatoire Othello ")

    while not done:
        env.render()
        legal = env.get_legal_actions()
        action = np.random.choice(legal)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.render()
    print("Reward final :", reward)

# def run_random_game():
#     """
#     Lance une partie simple avec actions aléatoires.
#     Sert à vérifier que l'environnement fonctionne.
#     """
#     env = TicTacToeEnv()
#     obs, _ = env.reset()
#     done = False

#     print("\n--- Partie aléatoire ---")

#     while not done:
#         env.render()

#         state = obs.flatten()
#         legal_actions = [i for i, v in enumerate(state) if v == 0]

#         action = np.random.choice(legal_actions)

#         obs, reward, terminated, truncated, _ = env.step(action)
#         done = terminated or truncated

#     env.render()
#     print("Reward final :", reward)


def run_mcts_test(iterations=50):
    """
    Test simple du MCTS pour vérifier que l'arbre fonctionne.
    """
    # env = TicTacToeEnv()
    env = OthelloEnv()
    obs, _ = env.reset()

    state = obs.flatten()
    # possible_actions = [i for i, v in enumerate(state) if v == 0]
    possible_actions = env.get_legal_actions()

    tree = MCTSTree(state, possible_actions)

    print("\n--- Test MCTS ---")

    for _ in range(iterations):

        node = tree.root

        # SELECTION
        while node.is_fully_expanded() and node.children:
            node = node.best_child()

        # EXPANSION
        if node.untried_actions:
            action = node.untried_actions[0]

            # env_copy = TicTacToeEnv()
            env_copy = OthelloEnv()
            env_copy.reset()
            # env_copy.board = node.state.reshape(3, 3).copy()
            env_copy.board = node.state.reshape(8, 8).copy()

            obs, reward, terminated, truncated, _ = env_copy.step(action)

            new_state = obs.flatten()
            # new_actions = [i for i, v in enumerate(new_state) if v == 0]
            new_actions = env_copy.get_legal_actions()

            node = node.add_child(new_state, action, new_actions)

        # SIMULATION
        reward = rollout(env, node.state)

        # BACKPROPAGATION
        backpropagate(node, reward)

    print("Taille arbre :", tree.tree_size())
    print("Profondeur arbre :", tree.max_depth())


def run_dqn_training():
    """
    Lance l'entraînement du DQN.
    """
    print("\n--- Lancement entraînement DQN ---")
    train_dqn()


def menu():
    print("\n====== PROJET RL ======")
    print("1 - Tester environnement (random)")
    print("2 - Tester MCTS")
    print("3 - Entraîner DQN")
    print("4 - Quitter")


def main():
    while True:

        menu()
        choice = input("Choix : ")

        if choice == "1":
            run_random_game()

        elif choice == "2":
            run_mcts_test()

        elif choice == "3":
            run_dqn_training()

        elif choice == "4":
            break

        else:
            print("Choix invalide")


if __name__ == "__main__":
    main()
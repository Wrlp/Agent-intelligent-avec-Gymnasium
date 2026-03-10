import numpy as np

# ENVIRONNEMENT
from envs.game_env import TicTacToeEnv

# DQN
from dqn.train import train as train_dqn
from dqn.agent import DQNAgent
import os

# MCTS
from mcts.tree import MCTSTree
from mcts.mcts_agent import rollout, backpropagate
from copy import deepcopy


def run_random_game():
    """
    Lance une partie simple avec actions aléatoires.
    Sert à vérifier que l'environnement fonctionne.
    """
    env = TicTacToeEnv()
    obs, _ = env.reset()
    done = False

    print("\n--- Partie aléatoire ---")

    while not done:
        env.render()

        state = obs.flatten()
        legal_actions = [i for i, v in enumerate(state) if v == 0]

        action = np.random.choice(legal_actions)

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.render()
    print("Reward final :", reward)


def run_mcts_test(iterations=50):
    """
    Test simple du MCTS pour vérifier que l'arbre fonctionne.
    """
    env = TicTacToeEnv()
    obs, _ = env.reset()

    state = obs.flatten()
    possible_actions = [i for i, v in enumerate(state) if v == 0]

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

            env_copy = TicTacToeEnv()
            env_copy.reset()
            env_copy.board = node.state.reshape(3, 3).copy()

            obs, reward, terminated, truncated, _ = env_copy.step(action)

            new_state = obs.flatten()
            new_actions = [i for i, v in enumerate(new_state) if v == 0]

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


def run_dqn_game():
    """
    Lance une partie avec l'agent DQN entraîné.
    """
    env = TicTacToeEnv()
    obs, _ = env.reset()
    done = False

    state_dim = obs.flatten().shape[0]
    action_dim = 9

    agent = DQNAgent(state_dim, action_dim)

    model_path = os.path.join(os.path.dirname(__file__), "models/dqn_tictactoe_final.pth")
    if not os.path.exists(model_path):
        print(f"Aucun modèle trouvé à {model_path}. Lancez d'abord l'entraînement DQN.")
        return

    agent.load(model_path)
    agent.epsilon = 0.0
    print(f"Modèle chargé depuis {model_path}")

    print("\n--- Partie DQN ---")
    env.render()

    while not done:
        state = obs.flatten()
        legal_actions = [i for i, v in enumerate(state) if v == 0]

        action = agent.select_action(state, legal_actions)

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        env.render()

    print("Reward final :", reward)

def run_mcts_game(iterations=200):
    """
    Joue une partie complète avec MCTS et retourne le résultat.
    """
    env = TicTacToeEnv()
    obs, _ = env.reset()
    done = False
    reward = 0

    while not done:
        state = obs.flatten()
        possible_actions = [i for i, v in enumerate(state) if v == 0]

        if not possible_actions:
            break

        # Construire l'arbre MCTS depuis l'état actuel
        tree = MCTSTree(state, possible_actions)

        for _ in range(iterations):
            node = tree.root

            # SELECTION
            while node.is_fully_expanded() and node.children:
                node = node.best_child()

            # EXPANSION
            if node.untried_actions:
                action = node.untried_actions[0]
                env_copy = TicTacToeEnv()
                env_copy.reset()
                env_copy.board = node.state.reshape(3, 3).copy()

                obs_copy, r, term, trunc, _ = env_copy.step(action)
                new_state = obs_copy.flatten()
                new_actions = [i for i, v in enumerate(new_state) if v == 0]
                node = node.add_child(new_state, action, new_actions)

            # SIMULATION
            sim_reward = rollout(env, node.state)

            # BACKPROPAGATION
            backpropagate(node, sim_reward)

        # Choisir la meilleure action
        best = tree.root.most_visited_child()
        if best is None:
            break
        action = best.action

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    return reward


def run_comparison(num_games=100, mcts_iterations=200):
    """
    Compare DQN vs MCTS sur un nombre de parties définies.
    Chaque agent joue num_games parties et on compare les résultats.
    """
    print(f"  Comparaison DQN vs MCTS ({num_games} parties chacun)")
    

    # --- MCTS ---
    print(f"\n[MCTS] Simulation de {num_games} parties ({mcts_iterations} itérations/coup)...")
    mcts_wins = 0
    mcts_draws = 0
    mcts_losses = 0

    for i in range(num_games):
        try : 
            reward = run_mcts_game(iterations=mcts_iterations)
            if reward == 1:
                mcts_wins += 1
            elif reward == 0:
                mcts_draws += 1
            else:
                mcts_losses += 1
        except Exception as e:
            print(f"Erreur partie MCTS {i+1}: {e}")
            mcts_losses += 1

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{num_games} parties jouées...")
    # --- DQN ---
    print(f"[DQN]  Simulation de {num_games} parties...")
    env = TicTacToeEnv()
    obs, _ = env.reset()
    state_dim = obs.flatten().shape[0]
    action_dim = 9

    agent = DQNAgent(state_dim, action_dim)
    model_path = os.path.join(os.path.dirname(__file__), "models/dqn_tictactoe_final.pth")

    if not os.path.exists(model_path):
        print("Aucun modèle DQN trouvé. Lancez d'abord l'entraînement (option 3).")
        return

    agent.load(model_path)
    agent.epsilon = 0.0  # Mode exploitation pure

    dqn_wins = 0
    dqn_draws = 0
    dqn_losses = 0

    for i in range(num_games):
        obs, _ = env.reset()
        done = False
        reward = 0

        while not done:
            state = obs.flatten()
            legal_actions = [i for i, v in enumerate(state) if v == 0]
            if not legal_actions:
                break
            action = agent.select_action(state, legal_actions)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        if reward == 1:
            dqn_wins += 1
        elif reward == 0:
            dqn_draws += 1
        else:
            dqn_losses += 1

    # --- Affichage des résultats ---
    print(f"\n{'─'*50}")
    print(f"{'Méthode':<12} {'Victoires':>10} {'Nuls':>8} {'Défaites':>10} {'Win%':>8}")
    print(f"{'─'*50}")
    print(f"{'MCTS':<12} {mcts_wins:>10} {mcts_draws:>8} {mcts_losses:>10} {mcts_wins/num_games*100:>7.1f}%")
    print(f"{'DQN':<12} {dqn_wins:>10} {dqn_draws:>8} {dqn_losses:>10} {dqn_wins/num_games*100:>7.1f}%")
    print(f"{'─'*50}")

    # Verdict
    if mcts_wins > dqn_wins:
        print("\nMCTS performe mieux que DQN sur ce jeu.")
    elif dqn_wins > mcts_wins:
        print("\nDQN performe mieux que MCTS sur ce jeu.")
    else:
        print("\nLes deux méthodes sont à égalité.")



def menu():
    print("\n====== PROJET RL ======")
    print("1 - Tester environnement (random)")
    print("2 - Tester MCTS")
    print("3 - Entraîner DQN")
    print("4 - Jouer avec DQN")
    print("5 - Comparer MCTS vs DQN")   
    print("6 - Quitter") 


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
            run_dqn_game()

        elif choice == "5":
            run_comparison()

        elif choice == "6":
            break

        else:
            print("Choix invalide")


if __name__ == "__main__":
    main()
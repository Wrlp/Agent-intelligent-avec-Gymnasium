import numpy as np

# ENVIRONNEMENT
# from envs.game_env import TicTacToeEnv
from envs.game_env import OthelloEnv

# MCTS 
from mcts.tree import MCTSTree
from mcts.mcts_agent import rollout, backpropagate

# DQN
from dqn.train import train as train_dqn, normalize_obs, preprocess_pixels
from dqn.agent import DQNAgent

def run_dqn_vs_random_othello(model_path="models/dqn_othello_final.pth"):
    """
    Lance une partie d'Othello : DQN (Joueur 1) vs Aléatoire (Joueur 2).
    """
    import os
    import torch
    if not os.path.exists(model_path):
        print(f"Erreur : Le modèle {model_path} n'existe pas.")
        # Essayer de trouver un autre modèle dans le dossier models
        models = [f for f in os.listdir("models") if f.startswith("dqn_othello") and f.endswith(".pth")]
        if models:
            import re
            def extract_number(f):
                match = re.search(r'(\d+)', f)
                return int(match.group(1)) if match else 0
            models.sort(key=extract_number)
            model_path = os.path.join("models", models[-1])
            print(f"Utilisation du modèle alternatif : {model_path}")
        else:
            return

    # Détection automatique de la taille de l'état à partir du checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    # On regarde la taille du premier poids de convolution
    # OthelloCNN: [64, 1, 3, 3] | Atari CNNNetwork: [32, 1, 8, 8]
    first_weight = checkpoint['model_state_dict']['conv.0.weight']
    if first_weight.shape[2] == 8:
        print("Architecture détectée : Atari (84x84)")
        state_dim = (1, 84, 84)
        use_atari_arch = True
        try:
            env = OthelloEnv(mode="atari", render_mode="rgb_array")
            print("Environnement initialisé en mode Atari.")
        except Exception as e:
            print(f"Erreur initialisation Atari ({e}), repli sur mode Logic.")
            env = OthelloEnv(mode="logic")
    else:
        print("Architecture détectée : Othello Logic (8x8)")
        state_dim = (1, 8, 8)
        use_atari_arch = False
        env = OthelloEnv(mode="logic")

    action_dim = 65
    
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    agent.load(model_path)
    agent.epsilon = 0.0 

    obs, _ = env.reset()
    done = False
    
    print(f"\n--- DQN vs Aléatoire (Othello) ---")
    print(f"DQN : Noir (X), Aléatoire : Blanc (O)")

    while not done:
        env.render()
        current_player = env.current_player
        legal_actions = env.get_legal_actions()
        
        if current_player == 1:
            # Tour du DQN
            if use_atari_arch:
                state = preprocess_pixels(obs)
            else:
                state = normalize_obs(obs, current_player)
            
            action = agent.select_action(state, legal_actions=legal_actions)
            print(f"DQN choisit l'action : {action}")
        else:
            # Tour Aléatoire
            action = np.random.choice(legal_actions)
            print(f"Aléatoire choisit l'action : {action}")
            
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.render()
    score = env._score()
    print(f"Partie terminée !")
    print(f"Score final -> Noir (DQN): {score[1]}  Blanc (Aléatoire): {score[2]}")
    
    if score[1] > score[2]:
        print("Victoire du DQN !")
    elif score[2] > score[1]:
        print("Victoire de l'Aléatoire !")
    else:
        print("Match nul !")

def run_random_game():
    env = OthelloEnv(mode="logic")
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

def run_mcts_test(iterations=50):
    """
    Test simple du MCTS pour vérifier que l'arbre fonctionne.
    """
    # env = TicTacToeEnv()
    env = OthelloEnv(mode="logic")
    obs, _ = env.reset()

    state = obs.flatten()
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
            env_copy = env.clone()
            
            #env_copy.board = node.state.reshape(3, 3).copy()
            env_copy.set_state(node.state, env.current_player)

            obs, reward, terminated, truncated, _ = env_copy.step(action)

            new_state = obs.flatten()
            # new_actions = [i for i, v in enumerate(new_state) if v == 0]
            new_actions = env_copy.get_legal_actions()

            node = node.add_child(new_state, action, new_actions)

        # SIMULATION
        env_sim = env.clone()
        sim_reward = rollout(env_sim, node.state)

        # BACKPROPAGATION
        backpropagate(node, sim_reward)

    print("Taille arbre :", tree.tree_size())
    print("Profondeur arbre :", tree.max_depth())


def run_dqn_training():
    """
    Lance l'entraînement du DQN.
    """
    print("\n--- Lancement entraînement DQN ---")
    train_dqn()


def evaluate_dqn_othello(model_path="models/dqn_othello_final.pth", num_games=20):
    """
    Évalue les performances du DQN sur plusieurs parties sans affichage.
    """
    import os
    import torch
    if not os.path.exists(model_path):
        print(f"Modèle {model_path} introuvable.")
        return

    checkpoint = torch.load(model_path, map_location="cpu")
    first_weight = checkpoint['model_state_dict']['conv.0.weight']
    use_atari_arch = (first_weight.shape[2] == 8)
    state_dim = (1, 84, 84) if use_atari_arch else (1, 8, 8)

    if use_atari_arch:
        try:
            env = OthelloEnv(mode="atari", render_mode="rgb_array")
            print("Environnement évaluation initialisé en mode Atari.")
        except Exception as e:
            print(f"Erreur initialisation Atari évaluation ({e}), repli sur mode Logic.")
            env = OthelloEnv(mode="logic")
    else:
        env = OthelloEnv(mode="logic")

    agent = DQNAgent(state_dim=state_dim, action_dim=65)
    wins, draws, losses = 0, 0, 0

    print(f"\n--- Évaluation DQN ({num_games} parties) ---")
    print(f"Modèle : {model_path}")
    print(f"Architecture : {'Atari' if use_atari_arch else 'Logic'}")

    for i in range(num_games):
        obs, _ = env.reset()
        done = False
        # Le DQN joue alternativement Noir (1) et Blanc (2) pour une évaluation équilibrée
        dqn_player = 1 if i % 2 == 0 else 2

        while not done:
            current_player = env.current_player
            legal_actions = env.get_legal_actions()

            if current_player == dqn_player:
                state = preprocess_pixels(obs) if use_atari_arch else normalize_obs(obs, current_player)
                action = agent.select_action(state, legal_actions=legal_actions)
            else:
                action = np.random.choice(legal_actions)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        score = env._score()
        if score[1] == score[2]:
            draws += 1
        elif (score[1] > score[2] and dqn_player == 1) or (score[2] > score[1] and dqn_player == 2):
            wins += 1
        else:
            losses += 1

        if (i + 1) % 5 == 0:
            print(f"Parties jouées : {i+1}/{num_games} | Victoires : {wins}")

    win_rate = (wins / num_games) * 100
    print(f"\nRésultats finaux :")
    print(f"Victoires : {wins} ({win_rate:.1f}%)")
    print(f"Nuls      : {draws}")
    print(f"Défaites  : {losses}")
    return win_rate

def evaluate_mcts_othello(num_games=20, mcts_iter=100):
    """
    Évalue les performances du MCTS sur plusieurs parties contre un joueur aléatoire.
    """
    from mcts.tree import MCTSTree
    from mcts.mcts_agent import rollout, backpropagate

    env = OthelloEnv(mode="logic")

    wins, draws, losses = 0, 0, 0

    print(f"\n--- Évaluation MCTS ({num_games} parties) ---")
    print(f"Iterations MCTS par coup : {mcts_iter}")

    for i in range(num_games):
        obs, _ = env.reset()
        done = False

        # Alternance : MCTS joue noir puis blanc
        mcts_player = 1 if i % 2 == 0 else 2

        while not done:
            current_player = env.current_player
            legal_actions = env.get_legal_actions()

            if not legal_actions:
                break

            # ----- MCTS -----
            if current_player == mcts_player:
                state = obs.flatten()
                tree = MCTSTree(state, legal_actions)

                for _ in range(mcts_iter):
                    node = tree.root

                    # SELECTION
                    while node.is_fully_expanded() and node.children:
                        node = node.best_child()

                    # EXPANSION
                    if node.untried_actions:
                        action_exp = node.untried_actions[0]

                        env_copy = env.clone()
                        env_copy.set_state(node.state, env.current_player)

                        obs_copy, _, term, trunc, _ = env_copy.step(action_exp)

                        new_state = obs_copy.flatten()
                        new_actions = env_copy.get_legal_actions()

                        node = node.add_child(new_state, action_exp, new_actions)

                    # SIMULATION
                    env_sim = env.clone()
                    sim_reward = rollout(env_sim, node.state)

                    # BACKPROPAGATION
                    backpropagate(node, sim_reward)

                best = tree.root.most_visited_child()
                action = best.action if best else np.random.choice(legal_actions)

            # ----- Random -----
            else:
                action = np.random.choice(legal_actions)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        # Score final
        score = env._score()

        if score[1] == score[2]:
            draws += 1
        elif (score[1] > score[2] and mcts_player == 1) or (score[2] > score[1] and mcts_player == 2):
            wins += 1
        else:
            losses += 1

        if (i + 1) % 5 == 0:
            print(f"Parties jouées : {i+1}/{num_games} | Victoires : {wins}")

    win_rate = (wins / num_games) * 100

    print("\nRésultats finaux :")
    print(f"Victoires : {wins} ({win_rate:.1f}%)")
    print(f"Nuls      : {draws}")
    print(f"Défaites  : {losses}")

    return win_rate

def run_dqn_vs_mcts(model_path="models/dqn_othello_final.pth", num_games=10, mcts_iter=100):
    from mcts.tree import MCTSTree
    from mcts.mcts_agent import rollout, backpropagate
    import torch
    import os

    if not os.path.exists(model_path):
        print("Modèle introuvable.")
        return

    checkpoint = torch.load(model_path, map_location="cpu")
    first_weight = checkpoint['model_state_dict']['conv.0.weight']
    use_atari = (first_weight.shape[2] == 8)

    state_dim = (1, 84, 84) if use_atari else (1, 8, 8)
    env = OthelloEnv(mode="logic")

    agent = DQNAgent(state_dim=state_dim, action_dim=65)
    agent.load(model_path)
    agent.epsilon = 0.0

    dqn_wins, mcts_wins, draws = 0, 0, 0

    print(f"\n--- DQN vs MCTS ({num_games} parties) ---")

    for game in range(num_games):

        obs, _ = env.reset()
        done = False

        while not done:
            current_player = env.current_player
            legal_actions = env.get_legal_actions()

            if not legal_actions:
                break

            # ----- DQN -----
            if current_player == 1:
                state = preprocess_pixels(obs) if use_atari else normalize_obs(obs, current_player)
                action = agent.select_action(state, legal_actions)

            # ----- MCTS -----
            else:
                state = obs.flatten()
                tree = MCTSTree(state, legal_actions)

                for _ in range(mcts_iter):
                    node = tree.root

                    while node.is_fully_expanded() and node.children:
                        node = node.best_child()

                    if node.untried_actions:
                        action_exp = node.untried_actions[0]
                        env_copy = env.clone()
                        env_copy.set_state(node.state, env.current_player)

                        obs_copy, _, term, trunc, _ = env_copy.step(action_exp)

                        new_state = obs_copy.flatten()
                        new_actions = env_copy.get_legal_actions()

                        node = node.add_child(new_state, action_exp, new_actions)

                    env_sim = env.clone()
                    sim_reward = rollout(env_sim, node.state)
                    backpropagate(node, sim_reward)

                best = tree.root.most_visited_child()
                action = best.action if best else np.random.choice(legal_actions)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        score = env._score()

        if score[1] > score[2]:
            dqn_wins += 1
        elif score[2] > score[1]:
            mcts_wins += 1
        else:
            draws += 1

        print(f"Partie {game+1}/{num_games} terminée")

    print("\nRésultats :")
    print(f"DQN   : {dqn_wins}")
    print(f"MCTS  : {mcts_wins}")
    print(f"Nuls  : {draws}")


def menu():
    print("\n====== PROJET RL ======")
    print("1 - Tester environnement (random)")
    print("2 - Tester MCTS")
    print("3 - Évaluer MCTS")
    print("4 - Entraîner DQN")
    print("5 - DQN vs Aléatoire (1 partie visuelle)")
    print("6 - Évaluer DQN (Win rate sur 20 parties)")
    print("7 - DQN vs MCTS")
    print("8 - Quitter")


def main():
    while True:

        menu()
        choice = input("Choix : ")

        if choice == "1":
            run_random_game()

        elif choice == "2":
            run_mcts_test()
        
        elif choice == "3":
            evaluate_mcts_othello()

        elif choice == "4":
            run_dqn_training()

        elif choice == "5":
            run_dqn_vs_random_othello()

        elif choice == "6":
            evaluate_dqn_othello()

        elif choice == "7":
            run_dqn_vs_mcts()

        elif choice == "8":
            break

        else:
            print("Choix invalide")


if __name__ == "__main__":
    main()
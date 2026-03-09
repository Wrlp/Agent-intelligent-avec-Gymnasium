import argparse
from dqn.train import train as train_dqn

def main():
    parser = argparse.ArgumentParser(description="Projet IA - Agent Intelligent avec Gymnasium")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Mode d'exécution")
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "mcts"], help="Algorithme à utiliser")
    
    args = parser.parse_args()
    
    if args.algo == "dqn":
        if args.mode == "train":
            train_dqn()
        else:
            print("Mode test pour DQN non encore implémenté.")
    elif args.algo == "mcts":
        print("MCTS non encore implémenté.")

if __name__ == "__main__":
    main()

import sys
import os
import numpy as np
import torch
from dqn.agent import DQNAgent
from envs.game_env import TicTacToeEnv

def train():
    # Paramètres d'entraînement
    episodes = 2000
    target_update_freq = 50
    save_freq = 500
    batch_size = 128
    lr = 0.001
    gamma = 0.95
    
    env = TicTacToeEnv()
    state_dim = 9 # (3x3 aplati)
    action_dim = 9 # 9 cases possibles
    
    agent = DQNAgent(
        state_dim=state_dim, 
        action_dim=action_dim, 
        lr=lr, 
        gamma=gamma, 
        batch_size=batch_size,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.999 # Diminution lente de l'exploration
    )
    
    print(f"--- Lancement de l'entraînement DQN sur {agent.device} ---")
    print(f"Environnement: Tic-Tac-Toe (Self-Play)")
    print(f"Paramètres: Episodes={episodes}, LR={lr}, Batch={batch_size}\n")
    
    rewards_history = []
    losses_history = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        state = obs.flatten()
        
        episode_reward = 0
        done = False
        
        while not done:
            # Récupérer les actions légales (cases vides)
            legal_actions = [i for i, val in enumerate(state) if val == 0]
            
            if not legal_actions: # Plateau plein (ne devrait pas arriver avec done=True)
                break
                
            action = agent.select_action(state, legal_actions=legal_actions)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            next_state = next_obs.flatten()
            done = terminated or truncated
            
            # Stocker la transition
            agent.memory.push(state, action, reward, next_state, done)
            
            # Apprentissage
            loss = agent.train_step()
            if loss:
                losses_history.append(loss)
                
            state = next_state
            episode_reward += reward
        
        agent.update_epsilon()
        
        if (episode + 1) % target_update_freq == 0:
            agent.sync_target_network()
            
        rewards_history.append(episode_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            avg_loss = np.mean(losses_history[-100:]) if losses_history else 0
            print(f"Ep {episode+1}/{episodes} | Eps: {agent.epsilon:.2f} | R-Avg: {avg_reward:.2f} | L-Avg: {avg_loss:.4f}")
            
        if (episode + 1) % save_freq == 0:
            os.makedirs("models", exist_ok=True)
            agent.save(f"models/dqn_tictactoe_{episode+1}.pth")

    print("\nEntraînement terminé avec succès.")
    os.makedirs("models", exist_ok=True)
    agent.save("models/dqn_tictactoe_final.pth")

if __name__ == "__main__":
    train()

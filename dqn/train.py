import sys
import os
import numpy as np
import torch
from dqn.agent import DQNAgent
from envs.game_env import OthelloEnv
import cv2

def preprocess_pixels(pixels):
    """
    Pré-traitement des pixels Atari ou du plateau : N&B, redimensionnement 84x84.
    """
    if pixels is None:
        return np.zeros((1, 84, 84), dtype=np.float32)
    
    if len(pixels.shape) == 2:
        resized = cv2.resize(pixels, (84, 84), interpolation=cv2.INTER_NEAREST)
        return resized[np.newaxis, :].astype(np.float32)
        
    if len(pixels.shape) == 3 and pixels.shape[2] == 3:
        gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized[np.newaxis, :].astype(np.float32) / 255.0
    
    resized = cv2.resize(pixels, (84, 84))
    if len(resized.shape) == 2:
        return resized[np.newaxis, :].astype(np.float32)
    return resized.transpose(2, 0, 1).astype(np.float32)

def normalize_obs(obs, player):
    """
    Normalise l'observation pour qu'elle soit invariante au joueur (Self-Play).
    Le joueur courant voit ses pièces comme 1 et celles de l'adversaire comme -1.
    """
    norm_obs = np.zeros_like(obs, dtype=np.float32)
    norm_obs[obs == player] = 1.0
    norm_obs[obs == (3 - player)] = -1.0
    return norm_obs[np.newaxis, :]

def train():
    episodes = 5000
    target_update_freq = 100
    save_freq = 500
    batch_size = 256
    lr = 1e-4
    gamma = 0.99
    
    try:
        env = OthelloEnv(mode="atari", render_mode="rgb_array")
        print("Mode Atari activé.")
        use_atari = True
        state_dim = (1, 84, 84)
    except Exception as e:
        print(f"Impossible de charger le mode Atari ({e}), repli sur le mode logic.")
        env = OthelloEnv(mode="logic")
        use_atari = False
        state_dim = (1, 8, 8)
    
    action_dim = 65  # 64 cases + 1 "passer"
    
    agent = DQNAgent(
        state_dim=state_dim, 
        action_dim=action_dim, 
        lr=lr, 
        gamma=gamma, 
        batch_size=batch_size,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9994, 
        self_play=True
    )
    
    print(f"--- Lancement de l'entraînement DQN sur {agent.device} ---")
    # print(f"Environnement: Tic-Tac-Toe (Self-Play)")
    print(f"Environnement: Othello")
    print(f"Paramètres: Episodes={episodes}, LR={lr}, Batch={batch_size}\n")
    
    rewards_history = []
    losses_history = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        current_player = env.current_player
        
        if use_atari:
            state = preprocess_pixels(obs)
        else:
            state = normalize_obs(obs, current_player)
        
        episode_reward = 0
        done = False
        
        while not done:
            legal_actions = env.get_legal_actions()
            if not legal_actions:
                break
                
            prev_player = env.current_player
            action = agent.select_action(state, legal_actions=legal_actions)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            next_player = env.current_player
            player_switched = (next_player != prev_player)
            
            if use_atari:
                next_state = preprocess_pixels(next_obs)
            else:
                next_state = normalize_obs(next_obs, next_player)
            
            done = terminated or truncated
            
            agent.memory.push(state, action, reward, next_state, done, player_switched)
            
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
            agent.save(f"models/dqn_othello_{episode+1}.pth")

    print("\nEntraînement terminé avec succès.")
    os.makedirs("models", exist_ok=True)
    agent.save("models/dqn_othello_final.pth")

if __name__ == "__main__":
    train()

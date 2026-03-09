import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from dqn.replay_buffer import ReplayBuffer

class MLPNetwork(nn.Module):
    """Réseau de neurones simple (Multi-Layer Perceptron)."""
    def __init__(self, state_dim, action_dim):
        super(MLPNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

class CNNNetwork(nn.Module):
    """Réseau de neurones convolutif (pour les images Atari)."""
    def __init__(self, input_shape, action_dim):
        super(CNNNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[1], 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[2], 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        
        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class DQNAgent:
    """Agent mettant en œuvre l'algorithme Deep Q-Learning."""
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, 
                 buffer_size=100000, batch_size=32, epsilon_start=1.0, 
                 epsilon_end=0.1, epsilon_decay=0.99995):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if isinstance(state_dim, int):
            self.policy_net = MLPNetwork(state_dim, action_dim).to(self.device)
            self.target_net = MLPNetwork(state_dim, action_dim).to(self.device)
        else:
            self.policy_net = CNNNetwork(state_dim, action_dim).to(self.device)
            self.target_net = CNNNetwork(state_dim, action_dim).to(self.device)
            
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)

    def select_action(self, state, legal_actions=None):
        """Choisit une action selon la politique epsilon-greedy."""
        if random.random() < self.epsilon:
            if legal_actions is not None:
                return random.choice(legal_actions)
            return random.randint(0, self.action_dim - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            
            if legal_actions is not None:
                mask = torch.full((self.action_dim,), float('-inf')).to(self.device)
                mask[legal_actions] = 0
                q_values = q_values + mask
            
            return q_values.argmax().item()

    def update_epsilon(self):
        """Diminue la valeur de epsilon pour réduire l'exploration au fil du temps."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train_step(self):
        """Effectue une étape d'optimisation sur un batch de transitions."""
        if len(self.memory) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions)
        
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
            
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()

    def sync_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

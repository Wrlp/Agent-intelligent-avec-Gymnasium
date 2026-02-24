import unittest
import torch
import numpy as np
from dqn.agent import DQNAgent

class TestDQN(unittest.TestCase):
    def test_agent_initialization_mlp(self):
        state_dim = 9
        action_dim = 9
        agent = DQNAgent(state_dim, action_dim)
        self.assertEqual(agent.action_dim, action_dim)
        self.assertIsInstance(agent.policy_net, torch.nn.Module)

    def test_agent_initialization_cnn(self):
        state_dim = (1, 8, 8)
        action_dim = 4
        agent = DQNAgent(state_dim, action_dim)
        self.assertEqual(agent.action_dim, action_dim)
        # On vérifie que c'est bien un CNN (indirectement par l'absence d'erreur)

    def test_select_action(self):
        state_dim = 4
        action_dim = 2
        agent = DQNAgent(state_dim, action_dim)
        state = np.random.rand(state_dim).astype(np.float32)
        action = agent.select_action(state)
        self.assertIn(action, [0, 1])

    def test_train_step(self):
        state_dim = 4
        action_dim = 2
        agent = DQNAgent(state_dim, action_dim, batch_size=2)
        
        # Remplir la mémoire
        for _ in range(5):
            s = np.random.rand(state_dim).astype(np.float32)
            a = 0
            r = 1.0
            ns = np.random.rand(state_dim).astype(np.float32)
            d = False
            agent.memory.push(s, a, r, ns, d)
            
        loss = agent.train_step()
        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)

if __name__ == '__main__':
    unittest.main()

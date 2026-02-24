import numpy as np
import random
from collections import deque

class ReplayBuffer:
    """Mémoire de l'agent pour stocker les transitions."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Ajoute une transition dans le buffer."""
        # On s'assure que les états sont bien des numpy arrays
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Tire un échantillon aléatoire de transitions."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.uint8)
        )

    def __len__(self):
        return len(self.buffer)

from agent.sumtree import SumTree
import random
import numpy as np

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) implementation.
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-6):
        """
        Initialize PrioritizedReplayBuffer with parameters:
        - capacity: maximum size of replay buffer
        - alpha: determines how much prioritization is used (0 = uniform, 1 = full prioritization)
        - beta: importance-sampling correction factor (0 = no correction, 1 = full correction)
        - beta_increment: how much to increase beta over time
        - epsilon: small value to avoid zero priorities
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0

    @property
    def maxlen(self):
        """Added for compatibility with the original deque implementation"""
        return self.capacity

    def add(self, state, action, reward, next_state, done):
        """Add new experience with maximum priority"""
        experience = (state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)

    def sample(self, batch_size):
        """Sample a batch of experiences based on their priorities"""
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size

        # Increase beta each time we sample
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            # Sample uniformly from each segment
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            idx, priority, experience = self.tree.get(s)
            indices.append(idx)
            priorities.append(priority)
            batch.append(experience)

        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        weights = (len(self.tree.data) * sampling_probabilities) ** (-self.beta)
        weights = weights / weights.max()  # Normalize weights

        # Unpack experiences
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            np.array(weights, dtype=np.float32)
        )

    def update_priorities(self, indices, errors):
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, errors):
            priority = (error + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        """Return the current size of the buffer"""
        return self.tree.n_entries
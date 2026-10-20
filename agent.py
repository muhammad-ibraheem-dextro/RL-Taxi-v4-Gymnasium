import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table ko zeros se initialize karna
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state, exploit_only=False):
        # Epsilon-greedy strategy
        if not exploit_only and np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_size) # Explore (Random action)
        return np.argmax(self.q_table[state, :])      # Exploit (Best action)

    def update(self, state, action, reward, next_state, done):
        # Bellman Equation se Q-table update karna
        best_next_action = np.argmax(self.q_table[next_state, :])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action] * (not done)
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error

    def decay_epsilon(self):
        # Exploration ko ahista ahista kam karna
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, filepath="q_table.npy"):
        np.save(filepath, self.q_table)

    def load_model(self, filepath="q_table.npy"):
        self.q_table = np.load(filepath)
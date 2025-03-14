# dqn_agent.py
import random
import numpy as np
import tensorflow as tf
from agent.replay_buffer import PrioritizedReplayBuffer


class DQNAgent:
    def __init__(self, state_size, action_size,
                learning_rate = 0.0001,
                gamma = 0.98,
                epsilon = 1.0,
                epsilon_decay = 0.992,
                epsilon_min = 0.01,
                replay_memory_size=20000,
                replay_start_size = 512,
                batch_size=512,
                load_file=None,
                per_alpha=0.6,    # How much prioritization to use (0 to 1)
                per_beta=0.4):    # Initial importance sampling correction (increases to 1)

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Experience replay memory
        self.memory = PrioritizedReplayBuffer(
            capacity=replay_memory_size,
            alpha=per_alpha,
            beta=per_beta
        )

        self.replay_start_size = replay_start_size
        self.batch_size = batch_size

        self.model = self._build_model()
        if load_file:
            self.load(load_file)

        self.target_model = self._build_model()
        self.update_target_network() #ensure that the two models are the same at the beggining

    def _build_model(self):
        """Builds the DQN model."""
        with tf.device('/GPU:0'):
            model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
                tf.keras.layers.LayerNormalization(), # Layer Normalization after activation
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.LayerNormalization(), # Layer Normalization after activation
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.LayerNormalization(), # Layer Normalization after activation
                tf.keras.layers.Dense(self.action_size, activation='linear')
            ])
            model.summary()
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def update_target_network(self):
        """Updates the target network with the weights of the main network."""
        self.target_model.set_weights(self.model.get_weights())

    def soft_update_target_network(self, tau=0.01):
        """Soft update model parameters: θ_target = τ*θ_online + (1-τ)*θ_target"""
        online_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = tau * online_weights[i] + (1 - tau) * target_weights[i]

        self.target_model.set_weights(target_weights)

    def remember(self, state, action, reward, next_state, done):
        """Stores an experience tuple in memory."""
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state, epsilon, available_actions):
        """Chooses an action based on epsilon-greedy policy and availability."""
        if np.random.rand() <= epsilon:
            #return random.randrange(self.action_size)
            action = random.choice(available_actions)  # Explore
            action_params = None
            return action, action_params
        else:
            # Act
            q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
            # Filter q_values based on available actions
            available_q_values = q_values[0][available_actions]
            action_index_in_available = np.argmax(available_q_values)
            action = available_actions[action_index_in_available]
            # if action == 0:  # tap_element
            #     action_params = {"locator": {"text": "Messages"}}
            # elif action == 1:  # input_text
            #     action_params = {"locator": {"class_name": "android.widget.EditText"}, "text_to_input": "hello"}
            # else:
            #     action_params = None
            action_params = None
            return action, action_params # Exploit

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)

        loss = 0
        if len(self.memory) > self.replay_start_size:
            loss += self.replay()  # Train agent with experience replay
            # Apply soft updates to target network instead of hard updates
            self.soft_update_target_network(tau=0.01)
        return loss

    def replay(self):
        """Train the agent with prioritized experience replay."""
        # Sample batch of experiences with priorities
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)

        # Predict Q-values for current states (batch prediction)
        q_values = self.model.predict(states, verbose=0)
        # Predict Q-values for next states using the **online model** (for action selection)
        next_q_values_online = self.model.predict(next_states, verbose=0) # Use online model here
        # Predict Q-values for next states using the **target model** (for action evaluation)
        next_q_values_target = self.target_model.predict(next_states, verbose=0) # Use target model here

        # **Double DQN Target Calculation:**
        # 1. Action selection using online network:
        best_actions_online = np.argmax(next_q_values_online, axis=1)
        # 2. Action evaluation using target network:
        target_values = next_q_values_target[np.arange(self.batch_size), best_actions_online]
        targets = rewards + self.gamma * target_values * (1 - dones)

        # Calculate TD errors for updating priorities
        td_errors = np.abs(targets - q_values[np.arange(self.batch_size), actions])

        # Update the Q-values for the taken actions
        target_q_values = q_values.copy()
        batch_indices = np.arange(self.batch_size)
        target_q_values[batch_indices, actions] = targets

        # Train the model with importance sampling weights
        history = self.model.fit(
            states,
            target_q_values,
            sample_weight=weights,  # Apply importance sampling weights
            epochs=1,
            verbose=0
        )

        # Update priorities in the replay buffer
        self.memory.update_priorities(indices, td_errors)

        return np.mean(history.history['loss'])

    def load(self, name):
        """Loads model weights."""
        self.model.load_weights(name)

    def save(self, name):
        """Saves model weights."""
        self.model.save_weights(name)

    def get_q_values_for_state(self, state):
        """Returns Q-values for a given state."""
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return q_values[0] # Return the Q-values array

    def get_target_q_values_for_state(self, state):
        """Returns target Q-values for a given state."""
        q_values = self.target_model.predict(np.expand_dims(state, axis=0), verbose=0)
        return q_values[0] # Return the Q-values array

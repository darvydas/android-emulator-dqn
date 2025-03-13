# dqn_agent.py
import random
import numpy as np
import tensorflow as tf
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size,
                    learning_rate = 0.0001,
                    gamma = 0.85,
                    epsilon = 1.0,
                    epsilon_decay = 0.992,
                    epsilon_min = 0.01,
                    replay_memory_size=20000,
                    replay_start_size = 512,
                    batch_size=512,
                    load_file=None):

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=replay_memory_size)  # Experience replay memory
        self.replay_start_size = replay_start_size
        self.batch_size = batch_size

        self.model = self._build_model()
        if load_file:
            self.load(load_file)
        # self.load('logs/20250306_104919_model_1_step_to_goal/20250306_114707dqn_model_episode_300.weights.h5')
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

    def remember(self, state, action, reward, next_state, done):
        """Stores an experience tuple in memory."""
        self.memory.append((state, action, reward, next_state, done))

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
        return loss

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)

        # Separate states, actions, rewards, next_states, dones from minibatch
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])

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

        # Update the Q-values for the taken actions
        target_q_values = q_values  # Start with current Q-values
        indices = np.arange(self.batch_size)
        target_q_values[[indices], [actions]] = targets

        # Train the model on the batch (batch training)
        history = self.model.fit(states, target_q_values, epochs=1, verbose=0)
        total_loss = np.mean(history.history['loss']) # calculate mean loss for the batch

        return total_loss

    # def replay(self):
    #     minibatch = random.sample(self.memory, self.batch_size)

    #     # Separate states, actions, rewards, next_states, dones from minibatch
    #     states = np.array([experience[0] for experience in minibatch])
    #     actions = np.array([experience[1] for experience in minibatch])
    #     rewards = np.array([experience[2] for experience in minibatch])
    #     next_states = np.array([experience[3] for experience in minibatch])
    #     dones = np.array([experience[4] for experience in minibatch])

    #     # Predict Q-values for current states (batch prediction)
    #     q_values = self.model.predict(states, verbose=0)
    #     # Predict Q-values for next states (batch prediction)
    #     next_q_values = self.target_model.predict(next_states, verbose=0)

    #     # Calculate target Q-values for each experience in the batch
    #     targets = rewards + self.gamma * np.amax(next_q_values, axis=1) * (1 - dones)

    #     # Update the Q-values for the taken actions
    #     target_q_values = q_values  # Start with current Q-values
    #     indices = np.arange(self.batch_size)
    #     target_q_values[[indices], [actions]] = targets

    #     # Train the model on the batch (batch training)
    #     history = self.model.fit(states, target_q_values, epochs=1, verbose=0)
    #     total_loss = np.mean(history.history['loss']) # calculate mean loss for the batch

    #     return total_loss

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

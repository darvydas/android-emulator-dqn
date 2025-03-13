# main.py
import datetime
import time
from android_env import AndroidEnv #, extract_features_from_tree
from dqn_agent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt
from logger.logger import Logger

from tools.graphviz import visualize_q_tree_graphviz

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))

if __name__ == "__main__":

    logger = Logger()

    # Appium Desired Capabilities (adjust to your emulator setup)
    desired_caps = {
        "platformName": "Android",
        "platformVersion": "15.0", # Adjust to your emulator version
        "deviceName": "Pixel_8_API_Vanilla_Hardware", # Or the name of your emulator
        "appPackage": "com.google.android.apps.nexuslauncher", # Default launcher package
        "appActivity": "com.google.android.apps.nexuslauncher.NexusLauncherActivity", # Default launcher activity
        "automationName": "UiAutomator2",
        "newCommandTimeout": 20,
        'chromedriverExecutable': '/path/to/chromedriver' # Optional: path to chromedriver if needed
    }

    # Initialize environment
    env = AndroidEnv(desired_caps, logger)
    # Initialize state and action size
    state_size = env.state_size
    action_size = env.action_size
    # Initialize DQN agent
    # agent = DQNAgent(state_size, action_size, load_file='logs/20250307_105733_model/20250307_151820dqn_model_episode_200.weights.h5')
    agent = DQNAgent(state_size, action_size)

    # Hyperparameters
    episodes = 1000  # Total number of training episodes
    target_update_interval = 10  # Episodes between target network updates (for stability)
    save_interval = 1  # Episodes between saving model weights and visualizations
    max_steps = 300  # Maximum steps per episode to prevent infinite loops

    cumulative_rewards = []
    episode_rewards = []
    avg_steps_plot = []
    accumulative_rewards_total = []

    # Training loop
    for episode in range(episodes):
        state = env.reset() # Reset environment at the start of each episode

        episode_reward = 0
        loss = 0
        step_count = 0
        done = False

        while not done and step_count < max_steps:
            available_actions = env.get_available_actions() # Environment returns available actions

            action, action_params = agent.act(state, agent.epsilon, available_actions) # agent filters actions from env

            next_state, reward, done, info = env.step(action, action_params)  # Environment steps

            # Agent store experience
            loss += agent.update(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            step_count += 1

        # Update target network periodically
        if (episode+1) % target_update_interval == 0:
            agent.update_target_network()

        # Print episode statistics
        avg_loss = loss / step_count if step_count > 0 else 0
        print(
            f"Episode: {episode + 1}/{episodes}, Reward: {episode_reward}, Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.2f}, Steps: {step_count}. Time since start: {logger.elapsed_time()}"
        )
        logger.append_move_log(f"Episode: {episode + 1}/{episodes}, Reward: {episode_reward}, Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.2f}, Steps: {step_count}. Time since start: {logger.elapsed_time()}")

        if episode_reward:
            cumulative_rewards.append(episode_reward)
            avg_steps_plot.append(step_count)
            accumulative_rewards_total.append(sum(cumulative_rewards))
        else:
            cumulative_rewards.append(0)
            avg_steps_plot.append(0)
            accumulative_rewards_total.append(0)


        if (episode+1) % save_interval == 0:
            logger.save_rewards_plot(cumulative_rewards,
                                    avg_steps_plot,
                                    accumulative_rewards_total,
                                    episode+1,
                                    {
                                        'learning_rate':agent.learning_rate,
                                        'gamma':agent.gamma,
                                        'epsilon':agent.epsilon,
                                        'epsilon_decay':agent.epsilon_decay,
                                        'epsilon_min':agent.epsilon_min,
                                        'replay_memory_size':agent.memory.maxlen,
                                        'replay_start_size':agent.replay_start_size,
                                        'batch_size':agent.batch_size
                                    }
                        )

            agent.save(f"{logger.model_directory}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}dqn_model_episode_{episode+1}.weights.h5")

            visualize_q_tree_graphviz(env.get_root_node(),
                                      agent.get_q_values_for_state,
                                      env.extract_features_from_tree,
                                      f"{logger.model_directory}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}q_value_tree_{episode + 1}")
            visualize_q_tree_graphviz(env.get_root_node(),
                                      agent.get_target_q_values_for_state,
                                      env.extract_features_from_tree,
                                      f"{logger.model_directory}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}target_q_value_tree_{episode + 1}")

        # Decay epsilon
        agent.decay_epsilon()

    env.close()  # Close the environment after training
    print(f"Training finished. Time since start: {logger.elapsed_time()}")
    logger.append_move_log(f"Training finished.")
    logger.close()
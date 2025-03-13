# README.md

## DQN Agent for Android Emulator Control

This project implements a Deep Q-Network (DQN) agent to control an Android emulator using Appium. The agent is trained to open the messaging app on the emulator's home screen.

### Setup Instructions

1.  **Install Appium and Appium-Python-Client:**

    ```bash
    npm install -g appium
    pip install Appium-Python-Client
    ```

2.  **Install ChromeDriver (Optional):**

    - If you encounter issues with web context or web elements, you might need to specify the path to `chromedriverExecutable` in the `desired_caps` in `main.py`. Download the ChromeDriver version compatible with your emulator's Chrome version and provide the path.

3.  **Set up Android Emulator:**

    - Create an Android emulator (e.g., Pixel 8) using Android Studio or other emulator tools.
    - Ensure the emulator is running and accessible.
    - Note the `platformVersion` and `deviceName` of your emulator to update `desired_caps` in `main.py`.

4.  **Install TensorFlow:**

    ```bash
    pip install tensorflow
    ```

5.  **Project Files:**
    - `android_env.py`: Contains the `AndroidEnv` class for interacting with the Android emulator.
    - `dqn_agent.py`: Contains the `DQNAgent` class implementing the DQN algorithm.
    - `main.py`: The main training script to run the DQN agent in the Android environment.
    - `README.md`: This file.

### Running the Training Script

1.  **Start Appium Server:**
    Open a terminal and start the Appium server:

    ```bash
    appium
    ```

2.  **Run `main.py`:**
    Open another terminal, navigate to the project directory, and run the training script:
    ```bash
    python main.py
    ```

### Code Explanation

- **`android_env.py`:**

  - Defines the `AndroidEnv` class to manage the Android emulator environment using Appium.
  - Provides methods for resetting the environment, taking actions, getting the state, and closing the connection.
  - The initial state is simplified to represent whether the messaging app is open or not.
  - Actions include tapping the messaging app icon and a NOOP action.
  - Reward system is designed to reward opening the messaging app.

- **`dqn_agent.py`:**

  - Defines the `DQNAgent` class implementing the Deep Q-Network algorithm.
  - Uses a simple MLP for the Q-Network built with TensorFlow/Keras.
  - Implements experience replay, epsilon-greedy action selection, and target network updates.
  - Provides methods to load and save the trained model weights.

- **`main.py`:**
  - Sets up Appium desired capabilities to connect to the emulator.
  - Initializes the `AndroidEnv` and `DQNAgent` classes.
  - Implements the main training loop, iterating through episodes and steps.
  - Performs action selection, environment interaction, experience replay, and target network updates.
  - Prints episode statistics and saves model weights periodically.

### Further Development

- **State Space Expansion:** Enhance the state representation to include more information about the emulator's UI (e.g., current activity, visible UI elements).
- **Action Space Expansion:** Add more actions to interact with different apps and UI elements (e.g., scrolling, text input).
- **Reward Function Refinement:** Design a more sophisticated reward function to guide the agent towards more complex tasks.
- **Network Architecture Tuning:** Experiment with different neural network architectures and hyperparameters for the DQN agent.
- **Parallel Execution:** Implement parallel training with multiple headless emulators for faster training.
- **Exploration Strategies:** Explore more advanced exploration techniques beyond epsilon-greedy.

**Note:** This is a basic starting point. You will likely need to adjust the code, especially the Appium desired capabilities, action space, reward function, and network architecture, based on your specific emulator setup and desired tasks. You may also need to adjust the XPath for finding the messaging app icon based on your emulator's UI layout.

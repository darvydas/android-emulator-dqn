# demo.py
import os
import time
import argparse
from android_env import AndroidEnv
from agent.dqn_agent import DQNAgent
from tools.dom_visualizer import DOMVisualizer
from logger.logger import Logger

def parse_args():
    parser = argparse.ArgumentParser(description='Run DQN Agent demo for Android UI navigation')
    parser.add_argument('--model', type=str, #required=True,
                        default='logs/20250315_141420_model_20250315_074841_epsil_0.40/20250316_011623dqn_model_episode_290.weights.h5',
                        help='Path to the saved model weights file (.h5)')
    parser.add_argument('--target_app', type=str, default="Messages",
                        help='Target app to navigate to (default: Messages)')
    parser.add_argument('--device_name', type=str, default="Pixel_8_API_Vanilla_Hardware",
                        help='Android device name')
    parser.add_argument('--platform_version', type=str, default="15.0",
                        help='Android platform version')
    parser.add_argument('--epsilon', type=float, default=0.0,
                        help='Epsilon value for exploration (0.0 = no exploration)')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between actions (seconds)')
    parser.add_argument('--max_steps', type=int, default=50,
                        help='Maximum number of steps to run the demo')
    return parser.parse_args()

def run_demo(args):
    print(f"Starting demo with model: {args.model}")
    print(f"Target app: {args.target_app}")
    print(f"Device: {args.device_name}, Android {args.platform_version}")

    # Set up logger
    logger = Logger()

    # Appium Desired Capabilities
    desired_caps = {
        "platformName": "Android",
        "platformVersion": args.platform_version,
        "deviceName": args.device_name,
        "appPackage": "com.google.android.apps.nexuslauncher",  # Default launcher package
        "appActivity": "com.google.android.apps.nexuslauncher.NexusLauncherActivity",
        "automationName": "UiAutomator2",
        "newCommandTimeout": 300,
        'chromedriverExecutable': '/path/to/chromedriver'  # Adjust as needed
    }

    # Initialize environment
    print("Initializing Android environment...")
    env = AndroidEnv(desired_caps, logger, target_apps=[args.target_app])

    # Get state and action sizes
    state_size = env.state_size
    action_size = env.action_size

    # Initialize DQN agent and load the pre-trained model
    print("Loading pre-trained DQN model...")
    agent = DQNAgent(state_size, action_size, load_file=args.model, epsilon=args.epsilon)

    # Initialize DOM visualizer
    print("Setting up UI visualizer...")
    visualizer = DOMVisualizer(env)

    # Reset the environment and get initial state
    print("Resetting environment...")
    state = env.reset() #target_app=args.target_app)
    # visualizer.update_status("RESET", 0)
    visualizer.update_and_draw()

    print(f"Starting navigation to {args.target_app}...")
    print("Press 'Space' to pause/continue, 'Q' to quit during visualization")

    # Run the demo
    episode_reward = 0
    visualizer.total_reward = 0
    step_count = 0
    done = False

    while not done and step_count < args.max_steps:
        # Get available actions from the environment
        available_actions = env.get_available_actions()

        # Agent selects an action
        action, action_params = agent.act(state, agent.epsilon, available_actions)
        action_name = env.action_space[action] if action < len(env.action_space) else "Unknown"

        print(f"Step {step_count + 1}: Taking action '{action_name}'")

        # Execute the action in the environment
        next_state, reward, done, info = env.step(action, action_params)

        # Update visualization
        visualizer.update_status(action, reward)
        # Update visualization and handle pausing
        viz_result = None
        while viz_result is None:  # Keep updating until we're no longer paused
            viz_result = visualizer.update_and_draw()
            if viz_result is False:  # Window was closed
                done = True
                break
            time.sleep(0.05)  # Small delay to prevent CPU hogging in the pause loop

        if not viz_result:  # Window was closed
            break


        # Update state and tracking variables
        state = next_state
        episode_reward += reward
        step_count += 1

        # Add a delay between steps for better visualization
        time.sleep(args.delay)

        if done:
            print(f"Navigation completed in {step_count} steps!")
            print(f"Total reward: {episode_reward}")
            # Keep the visualization open after completion
            visualizer.update_status("DONE", episode_reward)
            visualizer.update_and_draw()
            # Wait for user to close the visualization
            while visualizer.update_and_draw():
                time.sleep(0.1)

    if not done and step_count >= args.max_steps:
        print(f"Demo stopped after reaching maximum steps ({args.max_steps})")
        print(f"Total reward: {episode_reward}")

    # Clean up
    visualizer.close()
    env.close()
    logger.close()

    print("Demo completed!")

if __name__ == "__main__":
    args = parse_args()
    run_demo(args)
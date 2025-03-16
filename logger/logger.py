import datetime
import os
from matplotlib import pyplot as plt

LOGGER_ON = True

fig, ax = plt.subplots(figsize=(6, 4))

class Logger():
  def __init__(self, log_directory='logs/', move_log_directory='move_logs/'):
    self.start_time = datetime.datetime.now()
    self.timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
    self.log_file_name = f"log_{self.timestamp}.txt"
    self.log_directory = log_directory
    if not os.path.exists(self.log_directory):
      os.makedirs(self.log_directory)

    self.model_directory = self.log_directory + self.timestamp + '_model/'

    if not os.path.exists(self.model_directory):
      os.makedirs(self.model_directory)
    self.log_file_name = self.model_directory + self.log_file_name

    self.log_file = open(self.log_file_name, "w")

  def save_rewards_plot(self, cumulative_rewards, avg_steps_plot, accumulative_rewards_total, episode, model_hyperparameters):
    ax.clear()  # Clear the plot before redrawing

    # Create the combined plot (reward and average steps)
    ax.plot(cumulative_rewards, label="Episode Reward")
    # ax.plot(accumulative_rewards_total, label="Total acumulated rewards", color="orange")
    ax.plot(avg_steps_plot, label="Avg. Steps", color="red", linestyle="--")
    ax.axhline(y=0, color='black', linewidth=0.2)

    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward / Steps")  # Adjust y-label
    ax.set_title("Episode Rewards and Average Steps")
    # add hyperparameters on a plot as text
    textstr = '\n'.join((
        r'learning_rate=%.4f' % (model_hyperparameters['learning_rate'], ),
        r'gamma=%.2f' % (model_hyperparameters['gamma'], ),
        r'epsilon=%.3f' % (model_hyperparameters['epsilon'], ),
        r'epsilon_decay=%.3f' % (model_hyperparameters['epsilon_decay'], ),
        r'epsilon_min=%.2f' % (model_hyperparameters['epsilon_min'], ),
        r'replay_memory_size=%d' % (model_hyperparameters['replay_memory_size'], ),
        r'replay_start_size=%d' % (model_hyperparameters['replay_start_size'], ),
        r'batch_size=%d' % (model_hyperparameters['batch_size'], )))
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)

    ax.legend()  # Add a legend
    # fig.canvas.draw()
    plt.savefig(f"{self.model_directory}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}dqn_training_plot_episode_{episode}.png")

  def append_move_log(self, message):
    """Appends a message to the log file."""
    if not LOGGER_ON:
      return
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"{timestamp} - {message}\n"
    self.log_file.write(log_entry)
    self.log_file.flush()

  def elapsed_time(self):
    """Calculates and returns the elapsed time since the logger was initialized."""
    elapsed_time = datetime.datetime.now() - self.start_time

    days = elapsed_time.days
    seconds = elapsed_time.seconds
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    time_parts = []
    if days:
        time_parts.append(f"{days} days")
    if hours:
        time_parts.append(f"{hours} hours")
    if minutes:
        time_parts.append(f"{minutes} minutes")
    if seconds or not time_parts: # Always show seconds if no other units, or if seconds > 0
        time_parts.append(f"{seconds} seconds")

    return ", ".join(time_parts)


  def close(self):
    """Closes the log file."""
    self.log_file.close()
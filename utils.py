import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

# def validate_simulation(real_data, sim_data):
#     for param in ['speed', 'density']:
#         real = np.array([d[param] for d in real_data])
#         sim = np.array([d[param] for d in sim_data])
#         print(f"{param} D-statistic:", stats.ks_2samp(real, sim).statistic)

def plot_results(history):
    plt.figure(figsize=(10,5))
    plt.plot(history['train'], label='Training', marker='o')
    plt.plot(np.linspace(0, len(history['train'])-1, len(history['eval'])), 
             history['eval'], 'r-', label='Evaluation', marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('training_progress.png')
    plt.show()

def plot_speeds_across_episodes(speed_records):
    # Convert to 2D arrays: one per episode
    episode_speeds = [np.vstack(ep) for ep in speed_records]
    n_segments = len(episode_speeds[0][0])
    # n_eps = len(episode_speeds)

    for seg in range(n_segments):  # 6 segments
        plt.figure()
        for i, ep_log in enumerate(episode_speeds):
            speeds = [step_v[seg] for step_v in ep_log]  # Collect one segment across time
            plt.plot(speeds, label=f"Ep {i}")
        plt.title(f"Segment {seg+1} Speeds Across Valid Episodes")
        plt.xlabel("Timestep")
        plt.ylabel("Speed (km/h)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


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

def validate_simulation(real_data, sim_data):
    for param in ['speed', 'density']:
        real = np.array([d[param] for d in real_data])
        sim = np.array([d[param] for d in sim_data])
        print(f"{param} D-statistic:", stats.ks_2samp(real, sim).statistic)
import numpy as np
import matplotlib.pyplot as plt
from sim_env_full import MetaNetEnv

env = MetaNetEnv()
state = env.reset()
done = False

all_speeds = []
all_densities = []
vsl_applied = []

while not done:
    # Fixed VSL (no control)
    action = np.array([120.0, 120.0])
    vsl_applied.append(action.copy())

    state, reward, done, _ = env.step(action)
    all_speeds.append(np.array(env.v).flatten())
    all_densities.append(np.array(env.rho).flatten())

# Convert to arrays
all_speeds = np.array(all_speeds)       # shape: [timesteps, 6]
vsl_applied = np.array(vsl_applied)     # shape: [timesteps, 2]

# Plot segment-wise speeds
plt.figure(figsize=(10,5))
for i in range(all_speeds.shape[1]):
    plt.plot(all_speeds[:, i], label=f'Segment {i+1}')
plt.title("Segment Speeds Over Time (Fixed 120 km/h VSL)")
plt.xlabel("Timestep")
plt.ylabel("Speed (km/h)")
plt.legend()
plt.grid()
# plt.savefig('modelless_run.png')
plt.show()

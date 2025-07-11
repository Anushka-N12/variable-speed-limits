from turtle import speed
from sim_env import MetaNetEnv
from sim_env_mcity import MCityRoadEnv    # Import the specific environment
import matplotlib.pyplot as plt
import numpy as np

# Try different constant speed limits
test_speeds = [120, 100, 80, 60]
results = {}

for speed_limit in test_speeds:
    # env = MetaNetEnv()
    env = MCityRoadEnv()
    state = env.reset()
    rewards = []
    tts_over_time = []

    done = False
    while not done:
        action = np.array([speed_limit] * env.vsl_count)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        rho = np.sum(np.array(env.rho).flatten())
        tts_over_time.append(rho)

    total_reward = sum(rewards)
    tts = sum(tts_over_time) * env.T
    results[speed_limit] = {'reward': total_reward, 'tts': tts}

    print(f"Speed: {speed_limit}, Total Reward: {total_reward:.2f}, TTS: {tts:.2f}")

plt.figure()
plt.plot(list(results.keys()), [r['tts'] for r in results.values()], marker='o')
plt.xlabel("Speed Limit (km/h)")
plt.ylabel("Total Time Spent")
plt.title("Speed Limit vs TTS")
plt.grid(True)
plt.show()

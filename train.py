import os

# Set the environment variable to disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
from agent import ACAgent
from utils import plot_learning_curve
from sim_env import MetaNetEnv
import matplotlib.pyplot as plt

def evaluate(agent, env, n_episodes=3):
    """Proper evaluation function running complete episodes"""
    total = 0
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.choose_action(state, evaluate=True)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        total += episode_reward
    return total / n_episodes

def plot_results(history):
    plt.figure(figsize=(10,5))
    plt.plot(history['train'], label='Training')
    plt.plot(np.linspace(0, len(history['train'])-1, len(history['eval'])), 
             history['eval'], 'r-', label='Evaluation')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('training_progress.png')
    plt.show()

if __name__ == '__main__':
    # Initialize environment and agent
    env = MetaNetEnv()
    agent = ACAgent(
        min_s=60,
        max_s=120,
        input_dims=14,  # 2 speed limits + 6 speeds + 6 densities
        r_scale=0.01,
        tau=0.005,
        alpha=3e-4,
        gamma=0.99
    )
    
    # Training parameters
    n_eps = 12
    control_interval = 30  # Steps between speed limit changes
    history = {'train': [], 'eval': []}

    # Main training loop
    for ep in range(n_eps): 
        state = env.reset()
        done = False
        score = 0
        step = 0
        current_action = 120
        
        # Warmup network
        _ = agent.ac(tf.convert_to_tensor([state], dtype=tf.float32))
        nan_detected = False

        while not done:
            # Control logic
            if step % control_interval == 0:
                action = agent.choose_action(state)
                print(f"Action chosen: {action:.1f} at step {step}")
                prev_action = current_action
                current_action = action
            else:
                action = current_action
            
            # Environment step
            next_state, reward, done, _ = env.step(action)
            score += reward

            # Debug prints
            if step % 50 == 0:
                print(f"Step {step}: VSL {action:.1f}, Reward {reward:.2f}, "
                      # f"Speeds {np.array(env.v).flatten()[::2]}")
                      )

            # NaN detection
            if np.isnan(reward) or np.isnan(action):
                print(f"NaN detected at ep {ep}, step {step}!")
                nan_detected = True
                break
            
            # Store transition
            agent.memory.store_transition(
                np.array(state, dtype=np.float32),
                action,
                reward,
                np.array(next_state, dtype=np.float32),
                done
            )
            
            state = next_state
            step += 1
        
        # Learning and evaluation
        if not nan_detected:
            agent.learn()
            history['train'].append(score)

            # print(state, action, score, current_action)
            
            if ep % 5 == 0:
                eval_score = evaluate(agent, env)
                history['eval'].append(eval_score)
                print(f"Ep {ep}: Train {score:.1f} | Eval {eval_score:.1f} | VSL {current_action:.0f}")

        else:
            print(f"Skipping learning due to NaN in episode {ep}")

    # Final results
    plot_results(history)
    # plot_learning_curve(range(n_eps), history['train'], 'training_curve.png')

    # Only plot the episodes that actually completed:
    completed_episodes = len(history['train'])
    plot_learning_curve(range(completed_episodes), history['train'], 'training_curve.png')
import numpy as np
from agent import ACAgent
from utils import plot_learning_curve
from sim_env import MetaNetEnv
from replay_buffer import ReplayBuffer

if __name__ == '__main__':
    env = MetaNetEnv()
    agent = ACAgent(
        min_s=60,
        max_s=120,
        input_dims=[1],
        r_scale=1.0,
        tau=0.005,
        alpha=1e-5,
        gamma=0.99
    )
    n_eps = 1 #1800
    fname = 'vsl.png'
    figure_f = 'plots/' + fname
    best_score = -np.inf  # Might change with METANET
    score_history = []
    load_cp = False

    control_interval = 30  # Control interval for the agent
    prev_action = 0  # Previous action taken by the agent
    t = 0  # Time step counter

    # Load checkpoint if specified
    if load_cp:
        agent.load_models()  

    for i in range(n_eps):
        state = env.reset()
        done = False
        score = 0
        t = 0
        prev_action = 0  # or env.default_speed_limit

        while not done:
            t += 1

            # Choose action every 'control_interval' steps
            if t % control_interval == 0:
                action = agent.choose_action(state)
                prev_action = action
            else:
                action = prev_action

            next_state, reward, done, info = env.step(action)
            score += reward

            if not load_cp:
                agent.memory.store_transition(state, action, reward, next_state, done)

            state = next_state  # move to the next state

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            if not load_cp:
                agent.save_model()
        
        print(f"Episode {i}, score: {score:.2f}, avg_score: {avg_score:.2f}")

    x = [i+1 for i in range(n_eps)]
    plot_learning_curve(x, score_history, figure_f)
import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env = 'METANET'    # METANET
    agent = Agent(alpha=1e-5) #, n_actions=env.action_space.n)
    n_eps = 1800
    fname = 'vsl.png'
    figure_f = 'plots/' + fname
    best_score = env.reward_range[0]   # Might change with METANET
    score_history = []
    load_cp = False

    if load_cp:
        agent.load_models()

    for i in range(n_eps):
        observation = env.reset()      # Might change with METANET
        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation)
            n_observation, reward, done, info = env.step(action)       # Might change with METANET
            score += reward
            if not load_cp:
                agent.learn(observation, reward, n_observation, done)
                observation = n_observation
                score_history.append(score)
                avg_score = np.mean(score_history[-100:])
                if avg_score > best_score:
                    best_score = avg_score
                    if not load_cp:
                        agent.save_model()
                print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    x = [i+1 for i in range(n_eps)]
    plot_learning_curve(x, score_history, figure_f)
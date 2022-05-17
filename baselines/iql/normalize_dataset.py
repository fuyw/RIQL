import d4rl
import gym
import os
import numpy as np
import pandas as pd
from utils import ReplayBuffer


def get_minmax_traj_reward(env_name):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(obs_dim, act_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))

    # judge if it's a terminal state
    dones_float = np.zeros_like(replay_buffer.rewards)
    for i in range(len(dones_float) - 1):
        if np.linalg.norm(replay_buffer.observations[i+1] -
                          replay_buffer.next_observations[i]) > 1e-6 or replay_buffer.discounts[i] == 0.0:
            dones_float[i] = 1
    dones_float[-1] = 1

    # split trajectories
    trajs = [[]]
    for i in range(len(dones_float)):
        trajs[-1].append((replay_buffer.rewards[i].item()))
        if dones_float[i] == 1.0 and i + 1 < len(dones_float):
            trajs.append([])

    # find trajectory with min/max reward
    trajs.sort(key=sum)
    min_traj_reward = sum(trajs[0])
    max_traj_reward = sum(trajs[-1])
    return min_traj_reward, max_traj_reward


if __name__ == '__main__':
    os.makedirs('configs', exist_ok=True)
    res = []
    for env_name in [f'{i}-{j}-v2' for i in ['hopper', 'halfcheetah', 'walker2d'] for j in [
            'medium', 'medium-replay', 'medium-expert']]:
        min_traj_reward, max_traj_reward = get_minmax_traj_reward(env_name)
        res.append((env_name, min_traj_reward, max_traj_reward))
    res_df = pd.DataFrame(res, columns=['env_name', 'min_traj_reward', 'max_traj_reward'])
    res_df.to_csv('configs/minmax_traj_reward.csv')

import os
import time

import d4rl
import gym
import numpy as np
import pandas as pd
from tqdm import trange

from agents import (BCAgent, COMBOAgent, CQLAgent, CQLAgent2, IQLAgent,
                    TD3Agent, TD3BCAgent)
from utils import ReplayBuffer

###############
# Exp Setting #
###############
AGENTS = {
    "td3bc": TD3BCAgent,
    "cql": CQLAgent,
    "combo": COMBOAgent,
    "iql": IQLAgent,
    "td3": TD3Agent,
    "bc": BCAgent,
}
ALGOS = ["td3bc", "cql", "combo", "iql"]
ENVS = [
    "halfcheetah-medium-v2",
    "halfcheetah-medium-replay-v2",
    "halfcheetah-medium-expert-v2",
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
    "walker2d-medium-expert-v2",
]


##################
# Evaluate agent #
##################
def eval_policy(agent, algo, eval_env, mu, std, eval_episodes=10):
    t1 = time.time()
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = eval_env.reset(), False
        while not done:
            if algo == 'td3bc':
                obs = (obs - mu) / std
            action = agent.sample_action(agent.actor_state.params, obs)
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100
    return d4rl_score, time.time() - t1


#####################
# Sample trajectory #
#####################
def sample_trajectory(env_name):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load replay buffes ==> normalize obs for td3bc
    replay_buffer = ReplayBuffer(obs_dim, act_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    obs_mean, obs_std = replay_buffer.normalize_obs()

    # use agent ckpt to collect trajectories
    for algo in ALGOS:
        # load model ckpt
        ckpt_dirs = os.listdir(f"saved_models/{algo}_saved_models/{env_name}")
        for seed in range(5):
            agent = AGENTS[algo](obs_dim=obs_dim, act_dim=act_dim, seed=seed)
            ckpt_dir = f"saved_models/{algo}_saved_models/{env_name}/" + [
                i for i in ckpt_dirs if f"{algo}_s{seed}" in i
            ][0]

            untrained_eval_reward, _ = eval_policy(agent, algo, env, obs_mean,
                                                   obs_std)
            agent.load(ckpt_dir, step=200)
            eval_reward, _ = eval_policy(agent, algo, env, obs_mean, obs_std)
            print(
                f"Untrained {algo} agent with seed {seed}: eval_reward = {untrained_eval_reward:.2f}"
            )
            print(
                f"Ckpt {algo} agent with seed {seed}: eval_reward = {eval_reward:.2f}"
            )

            observations_, actions_, rewards_, next_observations_, dones_ = [], [], [], [], []
            while len(observations_) < 50000:
                obs, done = env.reset(), False
                while not done:
                    if algo == "td3bc":
                        normalized_obs = (obs - obs_mean) / obs_std
                    else:
                        normalized_obs = obs
                    action = agent.sample_action(agent.actor_state.params,
                                                 normalized_obs)
                    next_obs, reward, done, _ = env.step(action)
                    observations_.append(obs)
                    actions_.append(action)
                    rewards_.append(reward)
                    next_observations_.append(next_obs)
                    dones_.append(done)
                    obs = next_obs
            observations_ = np.array(observations_)
            actions_ = np.array(actions_)
            rewards_ = np.array(rewards_)
            next_observations_ = np.array(next_observations_)
            dones_ = np.array(dones_)
            np.savez(f"data/trajectories/{env_name}/{algo}_s{seed}",
                     observations=observations_,
                     actions=actions_,
                     rewards=rewards_,
                     next_observations=next_observations_,
                     dones=dones_)

    #################
    # optimal agent #
    #################
    optimal_agent = TD3Agent(obs_dim=obs_dim, act_dim=act_dim)
    untrained_eval_reward, _ = eval_policy(optimal_agent, "td3", env, obs_mean,
                                           obs_std)
    optimal_ckpt_file = f"saved_models/td3_saved_models/{env_name.split('-')[0]}-v2/optimal"
    optimal_agent.load(optimal_ckpt_file, step=0)
    eval_reward, _ = eval_policy(optimal_agent, "td3", env, obs_mean, obs_std)
    print(
        f"Untrained optimal agent: eval_reward = {untrained_eval_reward:.2f}")
    print(f"Ckpt optimal agent: eval_reward = {eval_reward:.2f}")

    # collect trajectories with optimal agent
    observations_, actions_, rewards_, next_observations_, dones_ = [], [], [], [], []
    while len(observations_) < 50000:
        obs, done = env.reset(), False
        while not done:
            action = optimal_agent.sample_action(
                optimal_agent.actor_state.params, obs)
            next_obs, reward, done, _ = env.step(action)
            observations_.append(obs)
            actions_.append(action)
            rewards_.append(reward)
            next_observations_.append(next_obs)
            dones_.append(done)
            obs = next_obs
    observations_ = np.array(observations_)
    actions_ = np.array(actions_)
    rewards_ = np.array(rewards_)
    next_observations_ = np.array(next_observations_)
    dones_ = np.array(dones_)
    np.savez(f"data/trajectories/{env_name}/optimal",
             observations=observations_,
             actions=actions_,
             rewards=rewards_,
             next_observations=next_observations_,
             dones=dones_)

    ############
    # bc agent #
    ############
    bc_agent = BCAgent(obs_dim=obs_dim, act_dim=act_dim)
    untrained_eval_reward, _ = eval_policy(bc_agent, "bc", env, obs_mean,
                                           obs_std)
    bc_agent.load(f"saved_models/bc_saved_models/{env_name}", step=10)
    eval_reward, _ = eval_policy(bc_agent, "bc", env, obs_mean, obs_std)
    print(f"Untrained bc agent: eval_reward = {untrained_eval_reward:.2f}")
    print(f"Ckpt bc agent: eval_reward = {eval_reward:.2f}")

    # collect trajectories with optimal agent
    observations_, actions_, rewards_, next_observations_, dones_ = [], [], [], [], []
    while len(observations_) < 50000:
        obs, done = env.reset(), False
        while not done:
            action = bc_agent.sample_action(bc_agent.actor_state.params, obs)
            next_obs, reward, done, _ = env.step(action)
            observations_.append(obs)
            actions_.append(action)
            rewards_.append(reward)
            next_observations_.append(next_obs)
            dones_.append(done)
            obs = next_obs
    observations_ = np.array(observations_)
    actions_ = np.array(actions_)
    rewards_ = np.array(rewards_)
    next_observations_ = np.array(next_observations_)
    dones_ = np.array(dones_)
    np.savez(f"data/trajectories/{env_name}/bc",
             observations=observations_,
             actions=actions_,
             rewards=rewards_,
             next_observations=next_observations_,
             dones=dones_)


if __name__ == "__main__":
    os.makedirs("data/trajectories", exist_ok=True)
    for env_name in ENVS:
        os.makedirs(f"data/trajectories/{env_name}", exist_ok=True)
        sample_trajectory(env_name)

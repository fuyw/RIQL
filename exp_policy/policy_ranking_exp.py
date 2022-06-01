import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import d4rl
import gym
import time

import numpy as np
import pandas as pd

from agents import (BCAgent, COMBOAgent, CQLAgent, CQLAgent2, IQLAgent,
                    TD3Agent, TD3BCAgent)
from utils import ReplayBuffer


######################
# Experiment Setting #
######################
AGENTS = {
    "iql": IQLAgent,
    "td3": TD3Agent,
    "td3bc": TD3BCAgent,
    "bc": BCAgent,
    "cql": CQLAgent,
    "combo": COMBOAgent,
}
ALGOS = ["td3bc", "combo", "cql", "iql", "bc", "optimal"]
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
CQL1_DICT = {
    "walker2d-medium-v2": [0, 1, 2],
    "hopper-medium-v2": [0, 1, 2],
    "halfcheetah-medium-v2": [0, 1, 2],
    "walker2d-medium-replay-v2": [],
    "hopper-medium-replay-v2": [],
    "halfcheetah-medium-replay-v2": [],
    "walker2d-medium-expert-v2": [],
    "hopper-medium-expert-v2": [],
    "halfcheetah-medium-expert-v2": []
}


###################
# Utils Functions #
###################
def eval_policy(agent, algo, eval_env, mu, std, eval_episodes=10):
    """Evaluate agent for 10 runs"""
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


def load_agents(env_name, env, obs_dim, act_dim, obs_mean, obs_std):
    """Load agents: optimal, bc, 4 baseline agents"""
    agent_dict = {}
    rew_res = []

    # load optimal agent
    optimal_agent = TD3Agent(obs_dim=obs_dim, act_dim=act_dim)
    untrained_eval_reward, _ = eval_policy(optimal_agent, "td3", env, obs_mean, obs_std, 3)
    optimal_ckpt_dir = f"saved_models/td3_saved_models/{env_name.split('-')[0]}-v2/optimal"
    optimal_agent.load(optimal_ckpt_dir, step=0)
    eval_reward, _ = eval_policy(optimal_agent, "td3", env, obs_mean, obs_std, 3)
    rew_res.append(("optimal", eval_reward))
    print(f"Untrained optimal agent: eval_reward = {untrained_eval_reward:.2f}")
    print(f"Ckpt optimal agent: eval_reward = {eval_reward:.2f}")

    # bc agent
    bc_agent = BCAgent(obs_dim=obs_dim, act_dim=act_dim)
    untrained_eval_reward, _ = eval_policy(bc_agent, "bc", env, obs_mean, obs_std, 3)
    bc_agent.load(f"saved_models/bc_saved_models/{env_name}", step=10)
    eval_reward, _ = eval_policy(bc_agent, "bc", env, obs_mean, obs_std, 3)
    rew_res.append(("bc", eval_reward))
    print(f"\nUntrained bc agent: eval_reward = {untrained_eval_reward:.2f}")
    print(f"Ckpt bc agent: eval_reward = {eval_reward:.2f}")

    for seed in range(5):
        agent_dict[f"optimal_s{seed}"] = optimal_agent
        agent_dict[f"bc_s{seed}"] = bc_agent

    for algo in ["td3bc", "combo", "cql", "iql"]:
        ckpt_dirs = os.listdir(f"saved_models/{algo}_saved_models/{env_name}")
        for seed in range(5):
            if algo == "cql" and seed in CQL1_DICT[env_name]:
                agent = CQLAgent2(obs_dim=obs_dim, act_dim=act_dim, seed=seed)
            else:
                agent = AGENTS[algo](obs_dim=obs_dim,
                                     act_dim=act_dim,
                                     seed=seed)
            ckpt_dir = f"saved_models/{algo}_saved_models/{env_name}/" + [
                i for i in ckpt_dirs if f"{algo}_s{seed}" in i
            ][0]
            untrained_eval_reward, _ = eval_policy(agent, algo, env, obs_mean,
                                                   obs_std, 3)
            agent.load(ckpt_dir, step=200)
            eval_reward, _ = eval_policy(agent, algo, env, obs_mean, obs_std,
                                         3)
            rew_res.append((f"{algo}_s{seed}", eval_reward))
            print(
                f"\nUntrained {algo} agent with seed {seed}: eval_reward = {untrained_eval_reward:.2f}"
            )
            print(
                f"Ckpt {algo} agent with seed {seed}: eval_reward = {eval_reward:.2f}"
            )
            agent_dict[f"{algo}_s{seed}"] = agent

    rew_res_df = pd.DataFrame(rew_res, columns=["algo", "reward"])
    return agent_dict, rew_res_df


##################
# Main Functions #
##################
def policy_action_q_mse(env_name="hopper-medium-v2"):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Normalize ReplayBuffer for TD3BC
    replay_buffer = ReplayBuffer(obs_dim, act_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    obs_mean, obs_std = replay_buffer.normalize_obs()

    # load agent ckpts
    agent_dicts, rew_df = load_agents(env_name, env, obs_dim, act_dim,
                                      obs_mean, obs_std)
    print(f"ckpt agent rewards:\n{rew_df}\n")

    # load behavior trajectory
    for seed in range(5):
        optimal_agent = agent_dicts[f"optimal_s{seed}"]
        seed_q, seed_mse = [], []
        for bc_algo in ["td3bc", "combo", "cql", "iql", "bc"]:
            # data collected by agent_seed
            if bc_algo in ["bc", "optimal"]:
                data = np.load(f"data/trajectories/{env_name}/{bc_algo}.npz")
            else:
                data = np.load(
                    f"data/trajectories/{env_name}/{bc_algo}_s{seed}.npz")

            # N states collected by the behavior agent
            bc_observations = data["observations"]  # (N, obs_dim)
            assert len(bc_observations) >= 50000
            bc_observations = bc_observations[:50000]

            # opitmal Q(s, a)
            optimal_actions = optimal_agent.sample_action(
                optimal_agent.actor_state.params, bc_observations)  # (2W, 3)

            # use agent ckpt to rerun the trajectory
            concat_qs, concat_actions, concat_mse = [], [], []
            for algo in ["td3bc", "combo", "cql", "iql", "bc"]:
                agent = agent_dicts[f"{algo}_s{seed}"]
                if algo == "td3bc":
                    normalized_obs = (bc_observations - obs_mean) / obs_std
                else:
                    normalized_obs = bc_observations
                agent_actions = agent.sample_action(agent.actor_state.params,
                                                    normalized_obs)
                agent_qs = optimal_agent.Q1(bc_observations, agent_actions)
                concat_qs.append(agent_qs.reshape(-1, 1))
                concat_actions.append(agent_actions.reshape(-1, 1, act_dim))

            # Use Q* to evaluate selected actions
            concat_qs = np.concatenate(concat_qs, axis=1)  # (2W, 5)
            concat_actions = np.concatenate(concat_actions,
                                            axis=1)  # (2W, 5, 3)

            # MSE actions
            for i in range(len(ALGOS[:-1])):
                agent_action_mse = np.sum(np.square(concat_actions[:, i, :] -
                                                    optimal_actions),
                                          axis=1)
                concat_mse.append(agent_action_mse.reshape(-1, 1))
            concat_mse = np.concatenate(concat_mse, axis=1)  # (2W, 5)

            seed_q.append(concat_qs)
            seed_mse.append(concat_mse)
        seed_q = np.concatenate(seed_q, axis=0)  # (100K, 5)
        seed_mse = np.concatenate(seed_mse, axis=0)  # (100K, 5)
        q_df = pd.DataFrame(seed_q, columns=ALGOS[:-1])
        mse_df = pd.DataFrame(seed_mse, columns=ALGOS[:-1])
        q_df.to_csv(f"res/policy_rank/{env_name}/q_s{seed}.csv")
        mse_df.to_csv(f"res/policy_rank/{env_name}/mse_s{seed}.csv")


if __name__ == "__main__":
    for env_name in ENVS:
        os.makedirs(f"res/policy_rank/{env_name}", exist_ok=True)
        policy_action_q_mse(env_name)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time

import d4rl
import gym
import numpy as np
import pandas as pd
from models import DynamicsModel
from collections import defaultdict
from tqdm import trange
from agents import (BCAgent, COMBOAgent, CQLAgent, CQLAgent2,
                    IQLAgent, TD3Agent, TD3BCAgent)
from utils import ReplayBuffer
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import top_k_accuracy_score

AGENTS = {
    "iql": IQLAgent,
    "td3": TD3Agent,
    "td3bc": TD3BCAgent,
    "bc": BCAgent,
    "cql": CQLAgent,
    "combo": COMBOAgent,
}
ALGOS = ["td3bc", "cql", "combo", "iql", "bc", "optimal"]
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

CQL1_DICT = {"walker2d-medium-v2": [0, 1, 2],
             "hopper-medium-v2": [0, 1, 2],
             "halfcheetah-medium-v2": [0, 1, 2],
             "walker2d-medium-replay-v2": [],
             "hopper-medium-replay-v2": [],
             "halfcheetah-medium-replay-v2": [],
             "walker2d-medium-expert-v2": [],
             "hopper-medium-expert-v2": [],
             "halfcheetah-medium-expert-v2": []}


# Evaluate agent for 10 runs
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


# Load agents: optimal, bc, 4 baseline agents
def load_agents(env_name, env, obs_dim, act_dim, obs_mean, obs_std):
    agent_dict = {}
    rew_res = []

    # load optimal agent
    optimal_agent = TD3Agent(obs_dim=obs_dim, act_dim=act_dim)
    untrained_eval_reward, _ = eval_policy(optimal_agent, "td3", env, obs_mean, obs_std, 3)
    optimal_ckpt_dir = f"saved_models/td3_saved_models/{env_name.split('-')[0]}-v2/optimal"
    optimal_agent.load(optimal_ckpt_dir, step=0)
    eval_reward, _ = eval_policy(optimal_agent, "td3", env, obs_mean, obs_std, 3)
    rew_res.append(("optimal", eval_reward))
    print(
        f"\nUntrained optimal agent: eval_reward = {untrained_eval_reward:.2f}"
    )
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

    for algo in ALGOS[:-2]:
        ckpt_dirs = os.listdir(f"saved_models/{algo}_saved_models/{env_name}")
        for seed in range(5):
            if algo == "cql" and seed in CQL1_DICT[env_name]:
                agent = CQLAgent2(obs_dim=obs_dim, act_dim=act_dim, seed=seed)
            else:
                agent = AGENTS[algo](obs_dim=obs_dim, act_dim=act_dim, seed=seed)            
            ckpt_dir = f"saved_models/{algo}_saved_models/{env_name}/" + [
                i for i in ckpt_dirs if f"{algo}_s{seed}" in i
            ][0]
            untrained_eval_reward, _ = eval_policy(agent, algo, env, obs_mean, obs_std, 3)
            agent.load(ckpt_dir, step=200)
            eval_reward, _ = eval_policy(agent, algo, env, obs_mean, obs_std, 3)
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


def run_ood_value_exp(env_name="hopper-medium-v2"):
    # initalize the environment
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Normalize ReplayBuffer for TD3BC
    replay_buffer = ReplayBuffer(obs_dim, act_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    obs_mean, obs_std = replay_buffer.normalize_obs()
    
    # probabilistic ensemble
    dynamics_model = DynamicsModel(env_name=env_name)
    dynamics_model.load(f"dynamics_models/{env_name}")

    # load agent ckpts
    agent_dicts, rew_df = load_agents(env_name, env, obs_dim, act_dim, obs_mean, obs_std)
    print(f"ckpt agent rewards:\n{rew_df}\n")

    # d4rl dataset
    ds = d4rl.qlearning_dataset(env)
    insample_std = dynamics_model.step(ds['observations'], ds['actions'])
    insample_std_90q = np.quantile(insample_std, 0.9)

    data = np.load(f"saved_buffers/{env_name.split('-')[0]}-v2/L100K.npz")
    observations = data["observations"]
    actions = data["actions"] 
    std = dynamics_model.step(observations, actions)  # N
    ood_mask = std > insample_std_90q

    res = []
    for algo in ALGOS: 
        if algo == "bc": continue
        if algo == "td3bc":
            normalized_obs = (observations - obs_mean) / obs_std
            normalized_d4rl_obs = (ds['observations'] - obs_mean) / obs_std
        else:
            normalized_obs = observations
            normalized_d4rl_obs = ds['observations']

        for seed in range(5):
            agent = agent_dicts[f"{algo}_s{seed}"]

            ood_Qs = agent.Q1(normalized_obs[ood_mask], actions[ood_mask])
            in_sample_Qs = agent.Q1(normalized_obs[~ood_mask], actions[~ood_mask])
            d4rl_Qs = agent.Q1(normalized_d4rl_obs, ds['actions'])

            ood_Qs_25q = np.quantile(ood_Qs, 0.25) 
            ood_Qs_50q = np.quantile(ood_Qs, 0.50) 
            ood_Qs_75q = np.quantile(ood_Qs, 0.75) 
            in_sample_Qs_25q = np.quantile(in_sample_Qs, 0.25)
            in_sample_Qs_50q = np.quantile(in_sample_Qs, 0.50)
            in_sample_Qs_75q = np.quantile(in_sample_Qs, 0.75)
            d4rl_Qs_25q = np.quantile(d4rl_Qs, 0.25)
            d4rl_Qs_50q = np.quantile(d4rl_Qs, 0.50)
            d4rl_Qs_75q = np.quantile(d4rl_Qs, 0.75)

            ratio_25q = ood_Qs_25q / d4rl_Qs_25q
            ratio_50q = ood_Qs_50q / d4rl_Qs_50q
            ratio_75q = ood_Qs_75q / d4rl_Qs_75q

            res.append((algo, seed, ood_Qs_25q, ood_Qs_50q, ood_Qs_75q,
                        in_sample_Qs_25q, in_sample_Qs_50q, in_sample_Qs_75q,
                        d4rl_Qs_25q, d4rl_Qs_50q, d4rl_Qs_75q,
                        ratio_25q, ratio_50q, ratio_75q))

    res_df = pd.DataFrame(res, columns=["algo", "seed", "ood_Qs_25q", "ood_Qs_50q", "ood_Qs_75q",
                                        "in_sample_Qs_25q", "in_sample_Qs_50q", "in_sample_Qs_75q",
                                        "d4rl_Qs_25q", "d4rl_Qs_50q", "d4rl_Qs_75q",
                                        "ratio_25q", "ratio_50q", "ratio_75q"])

    mu_res, std_res = [], []
    for algo in ALGOS[:4]:
        algo_df = res_df.query(f"algo == '{algo}'")
        mu_tmp, std_tmp = [algo], [algo]
        for col in ["ood_Qs_25q", "ood_Qs_50q", "ood_Qs_75q", "in_sample_Qs_25q",
                    "in_sample_Qs_50q", "in_sample_Qs_75q", "d4rl_Qs_25q",
                    "d4rl_Qs_50q", "d4rl_Qs_75q",
                    "ratio_25q", "ratio_50q", "ratio_75q"]:
            mu_tmp.append(algo_df[col].mean())
            std_tmp.append(algo_df[col].std())
        mu_res.append(mu_tmp)
        std_res.append(std_tmp)
    mu_res_df = pd.DataFrame(mu_res, columns=["algo", "ood_Qs_25q", "ood_Qs_50q", "ood_Qs_75q",
                                              "in_sample_Qs_25q", "in_sample_Qs_50q", "in_sample_Qs_75q",
                                              "d4rl_Qs_25q", "d4rl_Qs_50q", "d4rl_Qs_75q",
                                              "ratio_25q", "ratio_50q", "ratio_75q"])
    std_res_df = pd.DataFrame(std_res, columns=["algo", "ood_Qs_25q", "ood_Qs_50q", "ood_Qs_75q",
                                              "in_sample_Qs_25q", "in_sample_Qs_50q", "in_sample_Qs_75q",
                                              "d4rl_Qs_25q", "d4rl_Qs_50q", "d4rl_Qs_75q",
                                              "ratio_25q", "ratio_50q", "ratio_75q"])

    mu_res_df.to_csv(f"res/ood_value/{env_name}/mu_res.csv")
    std_res_df.to_csv(f"res/ood_value/{env_name}/std_res.csv")


if __name__ == "__main__":
    # for env_name in ENVS:
    #     os.makedirs(f"res/ood_value/{env_name}", exist_ok=True)
    #     os.makedirs(f"data/value_exp/state_actions/{env_name}", exist_ok=True)
    #     os.makedirs(f"data/value_exp/Qfunctions/{env_name}", exist_ok=True)
    #     os.makedirs(f"res/value_exp/rank_IC/{env_name}", exist_ok=True)

    for env_name in ENVS:
        # sample_different_actions(env_name)        # ==> collect (s, a) pair dataset
        # evaluate_Q_with_baseline_agent(env_name)  # ==> evaluate Q(s, a) values  # (N, 30)
        # compute_rank_IC(env_name)                 # ==> compute rank_IC
        run_ood_value_exp(env_name)

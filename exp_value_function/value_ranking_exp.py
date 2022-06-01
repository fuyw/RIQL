import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time

import d4rl
import gym
import numpy as np
import pandas as pd

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


# Preparing (s, a) pair dataset for value ranking experiment
def sample_different_actions(env_name="hopper-medium-v2"):
    # initalize the environment
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
        seed_observations, seed_actions = [], []
        for bc_algo in ALGOS:
            # data collected by agent_seed
            if bc_algo in ["bc", "optimal"]:
                data = np.load(f"data/trajectories/{env_name}/{bc_algo}.npz")
            else:
                data = np.load(f"data/trajectories/{env_name}/{bc_algo}_s{seed}.npz")

            # N states collected by the behavior agent
            bc_observations = data["observations"]  # (N, obs_dim)
            assert len(bc_observations) >= 50000
            bc_observations = bc_observations[:50000]

            # use agent ckpt to sample the action at each state
            concat_actions = []  # (N, 30)
            for algo in ALGOS:
                agent = agent_dicts[f"{algo}_s{seed}"]

                # normalize observations for the td3bc agent
                if algo == "td3bc":
                    normalized_obs = (bc_observations - obs_mean) / obs_std
                else:
                    normalized_obs = bc_observations

                # sample actions and add noises
                agent_actions = agent.sample_action(
                    agent.actor_state.params, normalized_obs).reshape(-1, 1, act_dim)
                action_noises = np.random.normal(size=(len(agent_actions), 5, act_dim)) * 0.2
                agent_actions += action_noises
        
                concat_actions.append(agent_actions.clip(-0.999999, 0.999999))

            # save (s, a) pair dataset for later use
            concat_actions = np.concatenate(concat_actions, axis=1)  # (N, 7, act_dim)

            # prepare to concat different agent's data
            seed_observations.append(bc_observations)  # (20000, 17)
            seed_actions.append(concat_actions)  # (20000, 30, 6)

        # save (s, a) for each seed
        seed_observations = np.concatenate(seed_observations, axis=0)  # (120000, 17)
        seed_actions = np.concatenate(seed_actions, axis=0)  # (120000, 30, 6)
        np.savez(f"data/value_exp/state_actions/{env_name}/s{seed}",
                 observations=seed_observations, actions=seed_actions)


# for each bc_algo & random seed we have a dataset
def evaluate_Q_with_baseline_agent(env_name="hopper-medium-v2"):
    # initalize the environment
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
        # load saved (s, a) dataset
        data = np.load(f"data/value_exp/state_actions/{env_name}/s{seed}.npz")
        bc_observations = data["observations"]  # (N, obs_dim)
        assert len(bc_observations) >= 50000
        bc_observations = bc_observations[:50000]
        sampled_actions = data["actions"][:50000]
        compare_num = sampled_actions.shape[1]  # M

        # use ckpt agents to evaluate Q(s, a)
        algo_qs_dict = {}
        for algo in ALGOS:
            if algo == "bc": continue
            agent = agent_dicts[f"{algo}_s{seed}"]

            # normalize observations for the td3bc agent
            if algo == "td3bc":
                normalized_obs = (bc_observations - obs_mean) / obs_std
            else:
                normalized_obs = bc_observations

            agent_Qs = []
            for i in range(compare_num):
                Qs = agent.Q1(normalized_obs, sampled_actions[:, i, :])
                agent_Qs.append(Qs.reshape(-1, 1))
            agent_Qs = np.concatenate(agent_Qs, axis=-1)
            algo_qs_dict[algo] = agent_Qs

        # save evaluated Q(s, a) result
        np.savez(f"data/value_exp/Qfunctions/{env_name}/s{seed}", **algo_qs_dict)


# compute rank accuracy based on the Q-functions
def compute_rank_IC(env_name="hopper-medium-v2"):
    for seed in range(5):
        # load saved (s, a) dataset ==> collected by `bc_algo` with `seed`
        data = np.load(f"data/value_exp/Qfunctions/{env_name}/s{seed}.npz")
        optimal_Qs = data["optimal"]  # (120000, 30) ==> 5*6=30
        best_actions = optimal_Qs.argmax(axis=1)

        # compute rank IC
        res = []
        for algo in ALGOS[:4]:
            algo_Qs = data[algo]  # (N, 30) ==> 5*6=30
            algo_rank_ICs = []
            for i in trange(len(algo_Qs)):
                rank_IC = spearmanr(optimal_Qs[i], algo_Qs[i])[0]
                algo_rank_ICs.append(rank_IC)
            avg_rank_IC = sum(algo_rank_ICs)/ len(algo_rank_ICs)
            top1_acc = top_k_accuracy_score(best_actions, algo_Qs, k=1)
            top3_acc = top_k_accuracy_score(best_actions, algo_Qs, k=3)
            top5_acc = top_k_accuracy_score(best_actions, algo_Qs, k=5)
            top10_acc = top_k_accuracy_score(best_actions, algo_Qs, k=10)
            res.append((algo, avg_rank_IC, top1_acc, top3_acc, top5_acc, top10_acc))
        res_df = pd.DataFrame(res, columns=["algo", "rank_IC", "top1_acc", "top3_acc", "top5_acc", "top10_acc"])

        # compute rank IC without optimal action
        res2 = []
        optimal_Qs2 = data["optimal"][:, :-5]  # (N, 25)
        best_actions2 = optimal_Qs2.argmax(axis=1)
        for algo in ALGOS[:4]:
            algo_Qs2 = data[algo][:, :-5]
            algo_rank_ICs2 = []
            for i in trange(len(algo_Qs2), desc=f"[s{seed}][Compute IC]"):
                rank_IC2 = spearmanr(optimal_Qs2[i], algo_Qs2[i])[0]
                algo_rank_ICs2.append(rank_IC2)
            avg_rank_IC2 = sum(algo_rank_ICs2)/ len(algo_rank_ICs2)
            top1_acc2 = top_k_accuracy_score(best_actions2, algo_Qs2, k=1)
            top3_acc2 = top_k_accuracy_score(best_actions2, algo_Qs2, k=3)
            top5_acc2 = top_k_accuracy_score(best_actions2, algo_Qs2, k=5)
            top10_acc2 = top_k_accuracy_score(best_actions2, algo_Qs2, k=10)
            res2.append((algo, avg_rank_IC2, top1_acc2, top3_acc2, top5_acc2, top10_acc2))
        res2_df = pd.DataFrame(res2, columns=["algo", "rank_IC2", "top1_acc2", "top3_acc2", "top5_acc2", "top10_acc2"])

        concat_df = pd.concat([res_df.set_index("algo"), res2_df.set_index("algo")], axis=1)
        concat_df.to_csv(f"res/value_exp/rank_IC/{env_name}/s{seed}.csv")


def summarize_result(env_name="hopper-medium-v2"):
    bc_algo_res = defaultdict(dict)
    for bc_algo in ALGOS:
        # store results for different random seeds for `bc_algo`
        rank_IC_res, top1_acc_res, top3_acc_res = defaultdict(list), defaultdict(list), defaultdict(list)
        rank_IC2_res, top1_acc2_res, top3_acc2_res = defaultdict(list), defaultdict(list), defaultdict(list)
        for seed in range(5):
            df = pd.read_csv(f"res/value_exp/rank_IC/{env_name}/{bc_algo}_s{seed}.csv", index_col=0)
            for algo in ALGOS[:4]:
                rank_IC_res[algo].append(df.loc[algo, 'rank_IC'])
                top1_acc_res[algo].append(df.loc[algo, 'top1_acc'])
                top3_acc_res[algo].append(df.loc[algo, 'top3_acc'])
                rank_IC2_res[algo].append(df.loc[algo, 'rank_IC2'])
                top1_acc2_res[algo].append(df.loc[algo, 'top1_acc2'])
                top3_acc2_res[algo].append(df.loc[algo, 'top3_acc2'])

        # compute mu & std
        mu_res = []
        std_res = []
        for algo in ALGOS[:4]:
            mu_rank_IC = np.array(rank_IC_res[algo]).mean()
            mu_top1_acc = np.array(top1_acc_res[algo]).mean()
            mu_top3_acc = np.array(top3_acc_res[algo]).mean()
            mu_rank_IC2 = np.array(rank_IC2_res[algo]).mean()
            mu_top1_acc2 = np.array(top1_acc2_res[algo]).mean()
            mu_top3_acc2 = np.array(top3_acc2_res[algo]).mean()
            mu_res.append((algo, mu_rank_IC, mu_top1_acc, mu_top3_acc, mu_rank_IC2, mu_top1_acc2, mu_top3_acc2))

        for algo in ALGOS[:4]:
            std_rank_IC = np.array(rank_IC_res[algo]).std()
            std_top1_acc = np.array(top1_acc_res[algo]).std()
            std_top3_acc = np.array(top3_acc_res[algo]).std()
            std_rank_IC2 = np.array(rank_IC2_res[algo]).std()
            std_top1_acc2 = np.array(top1_acc2_res[algo]).std()
            std_top3_acc2 = np.array(top3_acc2_res[algo]).std()
            std_res.append((algo, std_rank_IC, std_top1_acc, std_top3_acc, std_rank_IC2, std_top1_acc2, std_top3_acc2))

        # save as 2 dataframes
        mu_res_df = pd.DataFrame(mu_res, columns=['algo', 'rank_IC', 'top1_acc', 'top3_acc', 'rank_IC2', 'top1_acc2', 'top3_acc2'])
        std_res_df = pd.DataFrame(std_res, columns=['algo', 'rank_IC', 'top1_acc', 'top3_acc', 'rank_IC2', 'top1_acc2', 'top3_acc2'])
        bc_algo_res[bc_algo]['mu_res'] = mu_res_df.set_index('algo')
        bc_algo_res[bc_algo]['std_res'] = std_res_df.set_index('algo')

    col1, col2 = 'rank_IC', 'rank_IC2'
    for bc_algo in ALGOS:
        print_str = f"{bc_algo.upper()} "
        for algo in ALGOS[:4]:
            mu1 = bc_algo_res[bc_algo]['mu_res'].loc[algo, col1]
            std1 = bc_algo_res[bc_algo]['std_res'].loc[algo, col1]
            mu2 = bc_algo_res[bc_algo]['mu_res'].loc[algo, col2]
            std2 = bc_algo_res[bc_algo]['std_res'].loc[algo, col2]
            print_str += f"& {mu1:.2f} ({std1:.2f}) & {mu2:.2f} ({std2:.2f}) "
        print_str += "\\\\"
        print(print_str)
        print("\n")


    col1, col2 = 'top1_acc', 'top1_acc2'
    for bc_algo in ALGOS:
        print_str = f"{bc_algo.upper()} "
        for algo in ALGOS[:4]:
            mu1 = bc_algo_res[bc_algo]['mu_res'].loc[algo, col1]
            std1 = bc_algo_res[bc_algo]['std_res'].loc[algo, col1]
            mu2 = bc_algo_res[bc_algo]['mu_res'].loc[algo, col2]
            std2 = bc_algo_res[bc_algo]['std_res'].loc[algo, col2]
            print_str += f"& {mu1:.2f} ({std1:.2f}) & {mu2:.2f} ({std2:.2f}) "
        print_str += "\\\\"
        print(print_str)
        print("\n")


if __name__ == "__main__":
    for env_name in ENVS:
        os.makedirs(f"data/value_exp/state_actions/{env_name}", exist_ok=True)
        os.makedirs(f"data/value_exp/Qfunctions/{env_name}", exist_ok=True)
        os.makedirs(f"res/value_exp/rank_IC/{env_name}", exist_ok=True)

    for env_name in ENVS:
        sample_different_actions(env_name)        # ==> collect (s, a) pair dataset
        evaluate_Q_with_baseline_agent(env_name)  # ==> evaluate Q(s, a) values  # (N, 30)
        compute_rank_IC(env_name)                 # ==> compute rank_IC

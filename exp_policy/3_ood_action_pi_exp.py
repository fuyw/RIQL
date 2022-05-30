import os

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
import time

import d4rl
import gym
import numpy as np
import pandas as pd
from tqdm import trange

from agents import BCAgent
from models import DynamicsModel


######################
# Experiment Setting #
######################
ALGOS = ['offline data', 'td3bc', 'cql', 'combo', 'iql']
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


###################
# Utils Functions #
###################
def eval_policy(agent, eval_env, eval_episodes=10):
    """Evaluate agent for 10 runs"""
    t1 = time.time()
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = eval_env.reset(), False
        while not done:
            action = agent.sample_action(agent.actor_state.params, obs)
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_reward = eval_env.get_normalized_score(avg_reward) * 100.0
    return d4rl_reward, time.time() - t1


def get_insample_stats(env, bc_agent, dynamics_model):
    """get log_prob of the in-sample data"""
    ds = d4rl.qlearning_dataset(env)
    observations = ds['observations']
    actions = ds['actions']
    L = 10000
    num = len(observations) // L
    logps, mse, stds = [], [], []
    for i in trange(num):
        check_observations = observations[i * L:(i + 1) * L]
        check_actions = actions[i * L:(i + 1) * L]
        check_logps = bc_agent.get_logp(check_observations, check_actions)
        check_bc_actions = bc_agent.sample_action(bc_agent.actor_state.params,
                                                  check_observations)
        check_stds = dynamics_model.step(check_observations, check_actions)
        check_mse = np.square(check_actions - check_bc_actions).sum(1)
        logps.append(check_logps)
        mse.append(check_mse)
        stds.append(check_stds)
    logps = np.concatenate(logps)
    mse = np.concatenate(mse)
    stds = np.concatenate(stds)
    logp_10q = np.quantile(logps, 0.10)
    logp_25q = np.quantile(logps, 0.25)
    logp_50q = np.quantile(logps, 0.5)
    mse_10q = np.quantile(mse, 0.10)
    mse_25q = np.quantile(mse, 0.25)
    mse_50q = np.quantile(mse, 0.50)
    std_10q = np.quantile(stds, 0.10)
    std_25q = np.quantile(stds, 0.25)
    std_50q = np.quantile(stds, 0.50)
    return logp_10q, logp_25q, logp_50q, mse_10q, mse_25q, mse_50q, std_10q, std_25q, std_50q


##################
# Main Functions #
##################
def run_ood_action_exp():
    env_logp_mu_res, env_logp_std_res = [], []
    env_uncertainty_mu_res, env_uncertainty_std_res = [], []
    for env_name in ENVS:
        env = gym.make(env_name)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        bc_agent = BCAgent(obs_dim=obs_dim, act_dim=act_dim)
        bc_agent.load(f"saved_models/bc_saved_models/{env_name}", step=10)
        dynamics_model = DynamicsModel(env_name=env_name)
        dynamics_model.load(f"dynamics_models/{env_name}")

        # evaluate the behavior policy
        logp_10q, logp_25q, logp_50q, mse_10q, mse_25q, mse_50q, std_10q, std_25q, std_50q =\
            get_insample_stats(env, bc_agent, dynamics_model)

        # evaluate ckpt
        res = [("bc", 0, logp_10q, logp_25q, logp_50q, mse_10q, mse_25q,
                mse_50q, std_10q, std_25q, std_50q)]
        for algo in ["td3bc", "cql", "combo", "iql"]:
            for seed in range(5):
                data = np.load(f"data/trajectories/{env_name}/{algo}_s{seed}.npz")
                assert len(data["observations"]) >= 50000
                observations = data["observations"][:50000]
                actions = data["actions"][:50000]
                ckpt_logps, ckpt_mse, ckpt_stds = [], [], []
                for i in trange(1):
                    batch_observations = observations[:]
                    batch_actions = actions[:]
                    batch_logps = bc_agent.get_logp(batch_observations,
                                                    batch_actions)
                    batch_bc_actions = bc_agent.sample_action(
                        bc_agent.actor_state.params, batch_observations)
                    batch_mse = np.square(batch_actions -
                                          batch_bc_actions).sum(1)
                    batch_std = dynamics_model.step(batch_observations,
                                                    batch_actions)
                    ckpt_logps.append(batch_logps)
                    ckpt_mse.append(batch_mse)
                    ckpt_stds.append(batch_std)

                ckpt_logps = np.concatenate(ckpt_logps)
                ckpt_mse = np.concatenate(ckpt_mse)
                ckpt_stds = np.concatenate(ckpt_stds)
                ckpt_logp_10q = np.quantile(ckpt_logps, 0.10)
                ckpt_logp_25q = np.quantile(ckpt_logps, 0.25)
                ckpt_logp_50q = np.quantile(ckpt_logps, 0.50)
                ckpt_mse_10q = np.quantile(ckpt_mse, 0.10)
                ckpt_mse_25q = np.quantile(ckpt_mse, 0.25)
                ckpt_mse_50q = np.quantile(ckpt_mse, 0.50)
                ckpt_std_10q = np.quantile(ckpt_stds, 0.10)
                ckpt_std_25q = np.quantile(ckpt_stds, 0.25)
                ckpt_std_50q = np.quantile(ckpt_stds, 0.50)
                res.append(
                    (algo, seed, ckpt_logp_10q, ckpt_logp_25q, ckpt_logp_50q,
                     ckpt_mse_10q, ckpt_mse_25q, ckpt_mse_50q, ckpt_std_10q,
                     ckpt_std_25q, ckpt_std_50q))

        res_df = pd.DataFrame(res,
                              columns=[
                                  "algo", "seed", "logp_10q", "logp_25q",
                                  "logp_50q", "ckpt_mse_10q", "ckpt_mse_25q",
                                  "ckpt_mse_50q", "ckpt_std_10q",
                                  "ckpt_std_25q", "ckpt_std_50q"
                              ])
        res_df.iloc[:, -6:] *= 100.0

        # Prepare mu/std res
        mu_res, std_res = [], []
        for algo in ["bc", "td3bc", "cql", "combo", "iql"]:
            tmp_df = res_df.query(f"algo == '{algo}'").iloc[:, -9:]
            mu_res.append(tmp_df.mean(0))
            std_res.append(tmp_df.std(0))
        mu_res_df = pd.concat(mu_res, axis=1)
        std_res_df = pd.concat(std_res, axis=1)
        mu_res_df.columns = ["bc", "td3bc", "cql", "combo", "iql"]
        std_res_df.columns = ["bc", "td3bc", "cql", "combo", "iql"]

        env_logp_mu_res.append((env_name, *mu_res_df.loc['logp_50q'].values))
        env_logp_std_res.append((env_name, *std_res_df.loc['logp_50q'].values))
        env_uncertainty_mu_res.append(
            (env_name, *mu_res_df.loc['ckpt_std_50q'].values))
        env_uncertainty_std_res.append(
            (env_name, *std_res_df.loc['ckpt_std_50q'].values))

    logp_mu_df = pd.DataFrame(env_logp_mu_res, columns=['env_name', *ALGOS])
    logp_std_df = pd.DataFrame(env_logp_std_res, columns=['env_name', *ALGOS])
    uncertainty_mu_df = pd.DataFrame(env_uncertainty_mu_res,
                                     columns=['env_name', *ALGOS])
    uncertainty_std_df = pd.DataFrame(env_uncertainty_std_res,
                                      columns=['env_name', *ALGOS])

    logp_mu_df.to_csv("res/ood_action/logp_mu.csv")
    logp_std_df.to_csv("res/ood_action/logp_std.csv")
    uncertainty_mu_df.to_csv("res/ood_action/uncertainty_mu.csv")
    uncertainty_std_df.to_csv("res/ood_action/uncertainty_std.csv")


if __name__ == "__main__":
    os.makedirs("res/ood_action", exist_ok=True)
    run_ood_action_exp()

import os
import time

import d4rl
import gym
import numpy as np
import pandas as pd
from tqdm import trange

from agents import COMBOAgent, CQLAgent, IQLAgent, TD3Agent, TD3BCAgent
from models import ProbeTrainer
from utils import (ReplayBuffer, get_dot_product, get_kernel_norm,
                   get_optimal_actions, get_q_value, get_s_effective_dim,
                   get_s_embeddings, get_s_srank, get_sa_effective_dim,
                   get_sa_embeddings, get_sa_srank, load_data)

AGENTS = {
    "iql": IQLAgent,
    "td3": TD3Agent,
    "td3bc": TD3BCAgent,
    "cql": CQLAgent,
    "combo": COMBOAgent,
}
EPOCHS = 200
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


# Load optimal TD3 agent
def get_optimal_td3_agent(env_name, obs_dim, act_dim):
    optimal_agent = TD3Agent(obs_dim=obs_dim, act_dim=act_dim)
    optimal_agent.load(
        f"saved_models/optimal_models/{env_name.split('-')[0]}/", 100)
    return optimal_agent


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


# predict `r` based on 𝜙(s, a)
def probe_rewards(embeddings, rewards, epochs):
    trainer = ProbeTrainer(embeddings.shape[1], 1, epochs=epochs)
    kf_losses = trainer.train(embeddings, rewards.reshape(-1, 1))
    res = {f"cv_reward_loss{i+1}": loss for i, loss in enumerate(kf_losses)}
    return res


# predict `s'` based on 𝜙(s, a)
def probe_next_observations(embeddings, next_observations, epochs):
    trainer = ProbeTrainer(embeddings.shape[1],
                           next_observations.shape[1],
                           epochs=epochs)
    kf_losses = trainer.train(embeddings, next_observations)
    res = {f"cv_next_obs_loss{i+1}": loss for i, loss in enumerate(kf_losses)}
    return res


# predict `a` based on 𝜙(s, s')
def probe_inverse_actions(embeddings, actions, epochs):
    trainer = ProbeTrainer(embeddings.shape[1],
                           actions.shape[1],
                           epochs=epochs)
    kf_losses = trainer.train(embeddings, actions)
    res = {
        f"cv_inverse_action_loss{i+1}": loss
        for i, loss in enumerate(kf_losses)
    }
    return res


# predict `a*` based on 𝜙(s)
def probe_optimal_actions(embeddings, actions, epochs):
    trainer = ProbeTrainer(embeddings.shape[1],
                           actions.shape[1],
                           epochs=epochs)
    kf_losses = trainer.train(embeddings, actions)
    res = {
        f"cv_optimal_action_loss{i+1}": loss
        for i, loss in enumerate(kf_losses)
    }
    return res


# predict `Q*` based on  𝜙(s, a)
def probe_optimal_Qs(embeddings, optimal_Qs, epochs):
    trainer = ProbeTrainer(embeddings.shape[1], 1, epochs=epochs)
    kf_losses = trainer.train(embeddings, optimal_Qs.reshape(-1, 1))
    res = {f"cv_optimal_Q_loss{i+1}": loss for i, loss in enumerate(kf_losses)}
    return res


# predict `V*` based on 𝜙(s)
def probe_optimal_Vs(embeddings, optimal_Vs, epochs):
    trainer = ProbeTrainer(embeddings.shape[1], 1, epochs=epochs)
    kf_losses = trainer.train(embeddings, optimal_Vs.reshape(-1, 1))
    res = {f"cv_optimal_V_loss{i+1}": loss for i, loss in enumerate(kf_losses)}
    return res


# conduct 6 probing experiment for each ckpt
def get_ckpt_info(env, agent, algo, obs_mean, obs_std, ckpt_dir, step,
                  observations, actions, next_observations, fixed_idx):
    if step > 0: agent.load(ckpt_dir, step)
    sa_srank_info = get_sa_srank(agent, observations, actions)
    s_srank_info = get_s_srank(agent, observations)
    # s_effective_dim_info = get_s_effective_dim(agent, observations)
    # sa_effective_dim_info = get_sa_effective_dim(agent, observations, actions)
    dot_product_info = get_dot_product(agent, observations[fixed_idx],
                                       actions[fixed_idx],
                                       next_observations[fixed_idx])
    fixed_q = get_q_value(agent, observations[fixed_idx], actions[fixed_idx])
    kernel_norm = get_kernel_norm(
        agent.critic_state.params["critic1"]["net"]["Dense_1"]["kernel"])
    output_norm = get_kernel_norm(
        agent.critic_state.params["critic1"]["out_layer"]["kernel"])
    eval_reward, eval_time = eval_policy(agent, algo, env, obs_mean, obs_std,
                                         10)
    res = {
        **sa_srank_info,
        **s_srank_info,
        **dot_product_info,
        # **sa_effective_dim_info,
        # **s_effective_dim_info
    }
    res.update({
        "fixed_q": fixed_q.mean(),
        "fixed_q_max": fixed_q.max(),
        "fixed_q_min": fixed_q.min(),
        "fixed_q_std": fixed_q.std(),
        "kernel_norm": kernel_norm,
        "output_norm": output_norm,
        "eval_reward": eval_reward,
        "eval_time": eval_time,
        'step': step
    })
    return res


# fixed idx
fixed_idx = range(0, 100000, 5)


# experiment setting
def run_algo_exp(algo="td3bc"):
    for env_name in ENVS:
        env = gym.make(env_name)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        # load replay buffes
        replay_buffer = ReplayBuffer(obs_dim, act_dim)
        replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
        obs_mean, obs_std = replay_buffer.normalize_obs()

        # load testing data
        observations, actions, rewards, next_observations = load_data(env_name)

        # load optimal agent
        optimal_agent = get_optimal_td3_agent(env_name, obs_dim, act_dim)
        eval_reward, eval_time = eval_policy(optimal_agent, "td3", env,
                                             obs_mean, obs_std, 10)
        print(
            f"Optimal Agent on {env_name}: eval_reward = {eval_reward:.3f}, eval_time = {eval_time:.3f}"
        )
        optimal_Qs = get_q_value(optimal_agent, observations, actions)
        optimal_actions = get_optimal_actions(optimal_agent, observations)
        optimal_Vs = get_q_value(optimal_agent, observations, optimal_actions)

        # normalize the observations
        if algo == "td3bc":
            observations = (observations - obs_mean) / obs_std

        # load model ckpt
        ckpt_dirs = os.listdir(f"saved_models/{algo}_saved_models/{env_name}")
        for seed in range(5):
            # the result already exists
            if os.path.exists(f"res/{env_name}/{algo}/s{seed}.csv"):
                continue

            res = []
            agent = AGENTS[algo](obs_dim=obs_dim, act_dim=act_dim, seed=seed)

            ckpt_dir = f"saved_models/{algo}_saved_models/{env_name}/" + [
                i for i in ckpt_dirs if f"{algo}_s{seed}" in i
            ][0]
            for step in trange(0, 220, 20, desc=f"[{algo}][{env_name}][seed{seed}]"):
                step_res = get_ckpt_info(env, agent, algo, obs_mean, obs_std,
                                         ckpt_dir, step, observations, actions,
                                         next_observations, fixed_idx)

                # get embeddings
                sa_embeddings = get_sa_embeddings(agent, observations, actions)
                s_embeddings = get_s_embeddings(agent, observations)

                # prode reward
                probe_rewards_res = probe_rewards(sa_embeddings, rewards,
                                                  EPOCHS)
                step_res.update(probe_rewards_res)

                # probe next observations
                probe_next_obs_res = probe_next_observations(
                    sa_embeddings, next_observations, EPOCHS)
                step_res.update(probe_next_obs_res)

                # probe inverse actions
                if algo == "td3bc":
                    next_s_embeddings = get_s_embeddings(
                        agent,
                        (next_observations - obs_mean) / obs_std)  # (N, 256)
                else:
                    next_s_embeddings = get_s_embeddings(
                        agent, next_observations)  # (N, 256)
                ss_embeddings = np.concatenate(
                    [s_embeddings, next_s_embeddings], axis=-1)
                probe_inverse_action_res = probe_inverse_actions(
                    ss_embeddings, actions, EPOCHS)
                step_res.update(probe_inverse_action_res)

                # probe optimal actions
                probe_optimal_act_res = probe_optimal_actions(
                    s_embeddings, optimal_actions, EPOCHS)
                step_res.update(probe_optimal_act_res)

                # probe optimal Qs
                probe_optimal_q_res = probe_optimal_Qs(sa_embeddings,
                                                       optimal_Qs, EPOCHS)
                step_res.update(probe_optimal_q_res)

                # probe optimal Vs
                probe_optimal_v_res = probe_optimal_Vs(s_embeddings,
                                                       optimal_Vs, EPOCHS)
                step_res.update(probe_optimal_v_res)

                res.append(step_res)

            res_df = pd.DataFrame(res)
            res_df.to_csv(f"res/{env_name}/{algo}/s{seed}.csv")


# experiment setting
def run_exp(env_name="halfcheetah-medium-v2"):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load replay buffers
    replay_buffer = ReplayBuffer(obs_dim, act_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    obs_mean, obs_std = replay_buffer.normalize_obs()

    # load testing data
    observations, actions, rewards, next_observations = load_data(env_name)

    # load optimal agent
    optimal_agent = get_optimal_td3_agent(env_name, obs_dim, act_dim)
    eval_reward, eval_time = eval_policy(optimal_agent, "td3", env, obs_mean, obs_std, 10)
    print(f"Optimal Agent on {env_name}: eval_reward = {eval_reward:.3f}, eval_time = {eval_time:.3f}")
    optimal_Qs = get_q_value(optimal_agent, observations, actions)
    optimal_actions = get_optimal_actions(optimal_agent, observations)
    optimal_Vs = get_q_value(optimal_agent, observations, optimal_actions)

    for algo in ALGOS:
        # normalize the observations
        observations, actions, rewards, next_observations = load_data(env_name)
        if algo == "td3bc":
            observations = (observations - obs_mean) / obs_std

        # load model ckpt
        ckpt_dirs = os.listdir(f"saved_models/{algo}_saved_models/{env_name}")
        for seed in range(5):
            tmp_df = pd.read_csv(f"res/{env_name}/{algo}/s{seed}.csv")
            res = []
            agent = AGENTS[algo](obs_dim=obs_dim, act_dim=act_dim, seed=seed)

            ckpt_dir = f"saved_models/{algo}_saved_models/{env_name}/" + [
                i for i in ckpt_dirs if f"{algo}_s{seed}" in i][0]
            for step in trange(0, 220, 20, desc=f"[{algo}][{env_name}][seed{seed}]"):
                step_res = get_ckpt_info(env, agent, algo, obs_mean, obs_std,
                                         ckpt_dir, step, observations, actions,
                                         next_observations, fixed_idx)

                # get embeddings
                sa_embeddings = get_sa_embeddings(agent, observations, actions)
                s_embeddings = get_s_embeddings(agent, observations)

                # prode reward
                probe_rewards_res = probe_rewards(sa_embeddings, rewards, EPOCHS)
                step_res.update(probe_rewards_res)

                # probe next observations
                probe_next_obs_res = probe_next_observations(sa_embeddings, next_observations, EPOCHS)
                step_res.update(probe_next_obs_res)

                # probe inverse actions
                if algo == "td3bc":
                    next_s_embeddings = get_s_embeddings(
                        agent,
                        (next_observations - obs_mean) / obs_std)  # (N, 256)
                else:
                    next_s_embeddings = get_s_embeddings(
                        agent, next_observations)  # (N, 256)
                ss_embeddings = np.concatenate(
                    [s_embeddings, next_s_embeddings], axis=-1)
                probe_inverse_action_res = probe_inverse_actions(
                    ss_embeddings, actions, EPOCHS)
                step_res.update(probe_inverse_action_res)

                # probe optimal actions
                probe_optimal_act_res = probe_optimal_actions(
                    s_embeddings, optimal_actions, EPOCHS)
                step_res.update(probe_optimal_act_res)

                # probe optimal Qs
                probe_optimal_q_res = probe_optimal_Qs(sa_embeddings, optimal_Qs, EPOCHS)
                step_res.update(probe_optimal_q_res)

                # probe optimal Vs
                probe_optimal_v_res = probe_optimal_Vs(s_embeddings, optimal_Vs, EPOCHS)
                step_res.update(probe_optimal_v_res)

                res.append(step_res)

            res_df = pd.DataFrame(res)
            res_df.to_csv(f"res/{env_name}/{algo}/s{seed}.csv")


# experiment setting
def run_eff_dim_exp(algo="td3bc", L=5):
    # for env_name in ENVS:
    for env_name in ["halfcheetah-medium-v2"]:
        env = gym.make(env_name)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        # load replay buffers
        replay_buffer = ReplayBuffer(obs_dim, act_dim)
        replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
        obs_mean, obs_std = replay_buffer.normalize_obs()

        # load testing data
        observations, actions, _, _ = load_data(env_name)

        np.random.seed(100)
        random_idx = np.random.permutation(np.arange(len(actions)))[:L * 10000]

        # normalize the observations
        if algo == "td3bc":
            observations = (observations - obs_mean) / obs_std

        # load model ckpt
        ckpt_dirs = os.listdir(f"saved_models/{algo}_saved_models/{env_name}")
        for seed in range(5):
            res = []
            agent = AGENTS[algo](obs_dim=obs_dim, act_dim=act_dim, seed=seed)
            ckpt_dir = f"saved_models/{algo}_saved_models/{env_name}/" + [
                i for i in ckpt_dirs if f"{algo}_s{seed}" in i
            ][0]

            for step in trange(0, 220, 20, desc=f"[{algo}][{env_name}][seed{seed}]"):
                step_res = get_ckpt_info(agent, ckpt_dir, step, observations,
                                         actions, random_idx)
                res.append(step_res)

            res_df = pd.DataFrame(res)
            res_df.to_csv(f"res/{env_name}/{algo}/s{seed}_eff_dim_{L}W.csv")


if __name__ == "__main__":
    for env_name in ENVS:
        for algo in ALGOS:
            os.makedirs(f"res/{env_name}/{algo}", exist_ok=True)
    run_exp("halfcheetah-medium-v2")


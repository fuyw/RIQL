from typing import Tuple
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"
import ml_collections
import gym
import d4rl
import time
import pandas as pd
from tqdm import trange
from models import IQLAgent, NIQLAgent
from utils import ReplayBuffer, get_logger


AGENTS = {"iql": IQLAgent, "niql": NIQLAgent}


def normalize_rewards(replay_buffer: ReplayBuffer, env_name: str):
    if 'v2' in env_name:
        # mujoco environments
        normalize_info_df = pd.read_csv('configs/minmax_traj_reward.csv', index_col=0).set_index('env_name')
        min_traj_reward, max_traj_reward = normalize_info_df.loc[env_name, ['min_traj_reward', 'max_traj_reward']]
        replay_buffer.rewards = replay_buffer.rewards / (max_traj_reward - min_traj_reward) * 1000
    else:
        # antmaze environments
        replay_buffer.rewards -= 1.0


def eval_policy(agent: IQLAgent, env: gym.Env, eval_episodes: int = 10) -> Tuple[float, float]:
    t1 = time.time()
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = env.reset(), False
        while not done:
            action = agent.sample_action(agent.actor_state.params, obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_score = env.get_normalized_score(avg_reward) * 100
    return d4rl_score, time.time() - t1


def train_and_evaluate(configs: ml_collections.ConfigDict):
    start_time = time.time()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    exp_name = f'{configs.algo}_s{configs.seed}_{timestamp}'
    exp_info = f'# Running experiment for: {exp_name}_{configs.env_name} #'
    ckpt_dir = f"{configs.model_dir}/{configs.env_name}/{exp_name}"
    print('#'*len(exp_info) + f'\n{exp_info}\n' + '#'*len(exp_info))

    logger = get_logger(f'logs/{configs.env_name}/{exp_name}.log')
    logger.info(f"Exp configurations:\n{configs}")

    # initialize the d4rl environment 
    env = gym.make(configs.env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    agent = AGENTS[configs.algo](obs_dim=obs_dim,
                                 act_dim=act_dim,
                                 max_action=max_action,
                                 hidden_dims=configs.hidden_dims,
                                 seed=configs.seed,
                                 lr=configs.lr,
                                 tau=configs.tau,
                                 gamma=configs.gamma,
                                 expectile=configs.expectile,
                                 temperature=configs.temperature,
                                 max_timesteps=configs.max_timesteps,
                                 initializer=configs.initializer)

    # replay buffer
    replay_buffer = ReplayBuffer(obs_dim, act_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    normalize_rewards(replay_buffer, configs.env_name)

    logs = [{"step":0, "reward": eval_policy(agent, env, configs.eval_episodes)[0]}]

    for t in trange(1, configs.max_timesteps+1):
        batch = replay_buffer.sample(configs.batch_size)
        log_info = agent.update(batch)

        # Save every 1e5 steps & last 5 checkpoints
        if (t % 100000 == 0) or (t >= int(9.8e5) and t % 5000 == 0):
            agent.save(f"{ckpt_dir}", t // 5000)

        if (t>int(9.5e5) and (t % configs.eval_freq == 0)) or (t<=int(9.5e5) and t % (2*configs.eval_freq) == 0):
            eval_reward, eval_time = eval_policy(agent, env, configs.eval_episodes)
            log_info.update({
                "step": t,
                "reward": eval_reward,
                "eval_time": eval_time,
                "time": (time.time() - start_time) / 60
            })
            logs.append(log_info) 
            logger.info(
                f"\n[#Step {t}] eval_reward: {eval_reward:.3f}, eval_time: {eval_time:.3f}, time: {log_info['time']:.3f}\n"
                f"\tcritic_loss: {log_info['critic_loss']:.3f}, max_critic_loss: {log_info['max_critic_loss']:.3f}, min_critic_loss: {log_info['min_critic_loss']:.3f}\n"
                f"\tactor_loss: {log_info['actor_loss']:.3f}, max_actor_loss: {log_info['max_actor_loss']:.3f}, min_actor_loss: {log_info['min_actor_loss']:.3f}\n"
                # f"\tcontrastive_loss: {log_info['contrastive_loss']:.3f}, max_contrastive_loss: {log_info['max_contrastive_loss']:.3f}, min_contrastive_loss: {log_info['min_contrastive_loss']:.3f}\n"
                f"\tvalue_loss: {log_info['value_loss']:.3f}, max_value_loss: {log_info['max_value_loss']:.3f}, min_value_loss: {log_info['min_value_loss']:.3f}\n"
                f"\tweight: {log_info['weight']:.3f}, max_weight: {log_info['max_weight']:.3f}, min_weight: {log_info['min_weight']:.3f}\n"
                f"\tv: {log_info['v']:.3f}, max_v: {log_info['max_v']:.3f}, min_v: {log_info['min_v']:.3f}\n"
                # f"\texp_a: {log_info['exp_a']:.3f}, max_exp_a: {log_info['max_exp_a']:.3f}, min_exp_a: {log_info['min_exp_a']:.3f}\n"
                # f"\tadv: {log_info['adv']:.3f}, max_adv: {log_info['max_adv']:.3f}, min_adv: {log_info['min_adv']:.3f}\n"
                f"\tlog_prob: {log_info['log_prob']:.3f}, max_log_prob: {log_info['max_log_prob']:.3f}, min_log_prob: {log_info['min_log_prob']:.3f}\n"
                f"\tq1: {log_info['q1']:.3f}, max_q1: {log_info['max_q1']:.3f}, min_q1: {log_info['min_q1']:.3f}\n"
                f"\tq2: {log_info['q2']:.3f}, max_q2: {log_info['max_q2']:.3f}, min_q2: {log_info['min_q2']:.3f}\n"
                f"\ttarget_q: {log_info['target_q']:.3f}, max_target_q: {log_info['max_target_q']:.3f}, min_target_q: {log_info['min_target_q']:.3f}\n"
            )  

    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"{configs.log_dir}/{configs.env_name}/{exp_name}.csv")
    final_reward = log_df["reward"].iloc[-10:].mean()
    logger.info(f"\nAvg eval reward = {final_reward:.2f}\n")

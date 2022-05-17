from typing import Tuple
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"
import jax
import ml_collections
import gym
import d4rl
import time
import numpy as np
import pandas as pd
from tqdm import trange
from models import COMBOAgent, COMBOAgent_cqlreal, COMBOAgent_nocql, COMBOAgent_noimp
from utils import ReplayBuffer, get_logger


AGENTS = {"combo": COMBOAgent,
          "COMBOAgent_cqlreal": COMBOAgent_cqlreal,
          "COMBOAgent_nocql": COMBOAgent_nocql,
          "COMBOAgent_noimp": COMBOAgent_noimp}
def normalize_rewards(replay_buffer: ReplayBuffer, env_name: str):
    if 'maze' in env_name:
        replay_buffer.rewards = replay_buffer.rewards * 10.0 - 5.0


def eval_policy(agent, eval_env, eval_episodes: int = 10) -> Tuple[float, float]:
    t1 = time.time()
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = eval_env.reset(), False
        while not done:
            _, action = agent.eval_select_action(agent.actor_state.params, agent.rng, obs)
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100
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

    # Env parameters
    env = gym.make(configs.env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # set random seeds
    np.random.seed(configs.seed)
    env.seed(configs.seed)
    env.action_space.seed(configs.seed)

    # COMBO agent
    agent = AGENTS[configs.algo](env_name=configs.env_name,
                                 obs_dim=obs_dim,
                                 act_dim=act_dim,
                                 max_action=max_action,
                                 seed=configs.seed,
                                 tau=configs.tau,
                                 gamma=configs.gamma,
                                 noise_scale=configs.noise_scale,
                                 lr_critic=configs.lr_critic,
                                 lr_actor=configs.lr_actor,
                                 target_entropy=configs.target_entropy,
                                 backup_entropy=configs.backup_entropy,
                                 num_random=configs.num_random,
                                 min_q_weight=configs.min_q_weight,
                                 horizon=configs.horizon,
                                 real_ratio=configs.real_ratio,
                                 batch_size=configs.batch_size,
                                 rollout_batch_size=configs.rollout_batch_size,
                                 rollout_random=configs.rollout_random,
                                 hidden_dims=configs.hidden_dims,
                                 initializer=configs.initializer)
    # agent.model.train()

    # Load the trained dynamics model
    agent.model.load(f'saved_dynamics_models/{configs.env_name}')
    # agent.model.load(f'{configs.dynamics_model_dir}/{configs.env_name}/s0')
    logger.info(f"\nThe actor architecture is:\n{jax.tree_map(lambda x: x.shape, agent.actor_state.params)}")
    logger.info(f"\nThe critic architecture is:\n{jax.tree_map(lambda x: x.shape, agent.critic_state.params)}")

    # Replay buffer
    replay_buffer = ReplayBuffer(obs_dim, act_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    model_buffer = ReplayBuffer(obs_dim, act_dim, max_size=int(configs.horizon*1e5))
    normalize_rewards(replay_buffer, configs.env_name)

    # Evaluate the untrained policy
    logs = [{"step": 0, "reward": eval_policy(agent, env, configs.eval_episodes)[0]}]  # 2.38219
    logger.info(f"\n[# Step 0] eval_reward: {logs[0]['reward']:.2f}\n")
    for t in trange(1, configs.max_timesteps+1):
        log_info = agent.update(replay_buffer, model_buffer)

        # Save every 1e5 steps & last 5 checkpoints
        if (t % 100000 == 0) or (t >= int(9.8e5) and t % configs.eval_freq == 0):
            agent.save(f"{ckpt_dir}", t // configs.eval_freq)

        # save some evaluate time
        if (t>int(9.5e5) and (t % configs.eval_freq == 0)) or (t<=int(9.5e5) and t % (2*configs.eval_freq) == 0):
            eval_reward, eval_time = eval_policy(agent, env, configs.eval_episodes)
            log_info.update({"step": t, "reward": eval_reward, "eval_time": eval_time, "time": (time.time()-start_time)/60})
            logs.append(log_info)
            logger.info(
                f"\n[# Step {t}] eval_reward: {eval_reward:.2f}, eval_time: {eval_time:.2f}, time: {log_info['time']:.2f}\n"
                f"\talpha_loss: {log_info['alpha_loss']:.2f}, alpha: {log_info['alpha']:.2f}, logp: {log_info['logp']:.2f}\n"
                f"\tactor_loss: {log_info['actor_loss']:.2f}, sampled_q: {log_info['sampled_q']:.2f}\n"

                f"\tcritic_loss: {log_info['critic_loss']:.2f}, critic_loss_min: {log_info['critic_loss_min']:.2f}, "
                f"critic_loss_max: {log_info['critic_loss_max']:.2f}, critic_loss_std: {log_info['critic_loss_std']:.2f}\n"

                f"\tcritic_loss1: {log_info['critic_loss1']:.2f}, critic_loss1_min: {log_info['critic_loss1_min']:.2f}, "
                f"critic_loss1_max: {log_info['critic_loss1_max']:.2f}, critic_loss1_std: {log_info['critic_loss1_std']:.2f}\n"

                f"\tcritic_loss2: {log_info['critic_loss2']:.2f}, critic_loss2_min: {log_info['critic_loss2_min']:.2f}, "
                f"critic_loss2_max: {log_info['critic_loss2_max']:.2f}, critic_loss2_std: {log_info['critic_loss2_std']:.2f}\n"

                f"\treal_critic_loss: {log_info['real_critic_loss']:.2f}, fake_critic_loss: {log_info['fake_critic_loss']:.2f}\n"
                f"\treal_critic_loss_max: {log_info['real_critic_loss_max']:.2f}, fake_critic_loss_min: {log_info['fake_critic_loss_min']:.2f}\n"

                f"\tcql1_loss: {log_info['cql1_loss']:.2f}, cql1_loss_min: {log_info['cql1_loss_min']:.2f} "
                f"cql1_loss_max: {log_info['cql1_loss_max']:.2f}, cql1_loss_std: {log_info['cql1_loss_std']:.2f}\n"

                f"\tcql2_loss: {log_info['cql2_loss']:.2f}, cql2_loss_min: {log_info['cql2_loss_min']:.2f} "
                f"cql2_loss_max: {log_info['cql2_loss_max']:.2f}, cql2_loss_std: {log_info['cql2_loss_std']:.2f}\n"

                f"\ttarget_q: {log_info['target_q']:.2f}, target_q_min: {log_info['target_q_min']:.2f} "
                f"target_q_max: {log_info['target_q_max']:.2f}, target_q_std: {log_info['target_q_std']:.2f}\n"

                f"\tq1: {log_info['q1']:.2f}, q1_min: {log_info['q1_min']:.2f}, q1_max: {log_info['q1_max']:.2f}, q1_std: {log_info['q1_std']:.2f}\n"

                f"\tq2: {log_info['q2']:.2f}, q2_min: {log_info['q2_min']:.2f}, q2_max: {log_info['q2_max']:.2f}, q2_std: {log_info['q2_std']:.2f}\n"

                f"\tood_q1: {log_info['ood_q1']:.2f}, ood_q1_min: {log_info['ood_q1_min']:.2f}, "
                f"ood_q1_max: {log_info['ood_q1_max']:.2f}, ood_q1_std: {log_info['ood_q1_std']:.2f}\n"

                f"\tood_q2: {log_info['ood_q2']:.2f}, ood_q2_min: {log_info['ood_q2_min']:.2f}, "
                f"ood_q2_max: {log_info['ood_q2_max']:.2f}, ood_q2_std: {log_info['ood_q2_std']:.2f}\n"

                f"\tcql_q1: {log_info['cql_q1']:.2f}, cql_q2: {log_info['cql_q2']:.2f}, min_q_weight: {log_info['min_q_weight']:.1f}\n"
                f"\trandom_q1: {log_info['random_q1']:.2f}, random_q2: {log_info['random_q2']:.2f}, "
                f"logp_next_action: {log_info['logp_next_action']:.2f}\n"
                f"\treal_batch_rewards: {log_info['real_batch_rewards']:.2f}, real_batch_rewards_min: {log_info['real_batch_rewards_min']:.2f}, real_batch_rewards_max: {log_info['real_batch_rewards_max']:.2f}\n"
                f"\treal_batch_actions: {log_info['real_batch_actions']:.2f}, real_batch_observations: {log_info['real_batch_observations']:.2f}, "
                f"real_batch_discounts: {log_info['real_batch_discounts']:.2f}\n"
                f"\tmodel_batch_rewards: {log_info['model_batch_rewards']:.2f}, model_batch_rewards_min: {log_info['model_batch_rewards_min']:.2f}, model_batch_rewards_max: {log_info['model_batch_rewards_max']:.2f}\n"
                f"\tmodel_batch_actions: {log_info['model_batch_actions']:.2f}, model_batch_observations: {log_info['model_batch_observations']:.2f}, "
                f"model_batch_discounts: {log_info['model_batch_discounts']:.2f}\n"
                f"\tmodel_buffer_size: {log_info['model_buffer_size']:.0f}, "
                f"model_buffer_ptr: {log_info['model_buffer_ptr']:.0f}\n"
            )

    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"{configs.log_dir}/{configs.env_name}/{exp_name}.csv")

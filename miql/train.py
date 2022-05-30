import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

from typing import Tuple
import time
import d4rl
import gym
import jax
import ml_collections
import pandas as pd
from tqdm import trange

from models import MIQLAgent, MIQLAgent_CDA
from utils import ModelBuffer, ReplayBuffer, get_logger

AGENTS = {"miql": MIQLAgent, "cda": MIQLAgent_CDA}


def normalize_rewards(replay_buffer: ReplayBuffer, env_name: str):
    if 'v2' in env_name:
        # mujoco environments
        normalize_info_df = pd.read_csv('configs/minmax_traj_reward.csv',
                                        index_col=0).set_index('env_name')
        min_traj_reward, max_traj_reward = normalize_info_df.loc[
            env_name, ['min_traj_reward', 'max_traj_reward']]
        replay_buffer.rewards = replay_buffer.rewards / (
            max_traj_reward - min_traj_reward) * 1000
        replay_buffer.reward_delta = max_traj_reward - min_traj_reward
    else:
        # antmaze environments
        replay_buffer.rewards -= 1.0


def eval_policy(agent: MIQLAgent,
                env: gym.Env,
                eval_episodes: int = 10) -> Tuple[float, float]:
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
    batch_name = "".join([i[0] for i in configs.batchs.split('_')])
    exp_name = f'{configs.algo}_s{configs.seed}_ws{configs.warmup_timesteps//100000}_{batch_name}_an{configs.action_noise}_{timestamp}'
    exp_info = f'# Running experiment for: {exp_name}_{configs.env_name} #'
    ckpt_dir = f"{configs.model_dir}/{configs.env_name}/{exp_name}"
    print('#' * len(exp_info) + f'\n{exp_info}\n' + '#' * len(exp_info))

    logger = get_logger(f'logs/{configs.env_name}/{configs.algo}/{exp_name}.log')
    logger.info(f"Exp configurations:\n{configs}")

    # initialize the d4rl environment
    env = gym.make(configs.env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    agent = AGENTS[configs.algo](obs_dim=obs_dim,
                                 act_dim=act_dim,
                                 max_action=max_action,
                                 env_name=configs.env_name,
                                 hidden_dims=configs.hidden_dims,
                                 seed=configs.seed,
                                 lr=configs.lr,
                                 tau=configs.tau,
                                 gamma=configs.gamma,
                                 expectile=configs.expectile,
                                 temperature=configs.temperature,
                                 max_timesteps=configs.max_timesteps,
                                 horizon=configs.horizon,
                                 action_noise=configs.action_noise,
                                 warmup_timesteps=configs.warmup_timesteps,
                                 uncertainty=configs.uncertainty,
                                 eta=configs.eta,
                                 real_ratio=configs.real_ratio,
                                 quantile=configs.quantile,
                                 batchs=configs.batchs,
                                 initializer=configs.initializer)

    # Load the trained dynamics model
    # agent.model.train()
    agent.model.load(f'dynamics_models/{configs.env_name}')
    logger.info(
        f"\nThe actor architecture is:\n{jax.tree_map(lambda x: x.shape, agent.actor_state.params)}"
    )
    logger.info(
        f"\nThe critic architecture is:\n{jax.tree_map(lambda x: x.shape, agent.critic_state.params)}"
    )

    # replay buffer
    replay_buffer = ReplayBuffer(obs_dim, act_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    model_buffer = ModelBuffer(obs_dim,
                               act_dim,
                               max_size=int(configs.horizon * 1e5))
    normalize_rewards(replay_buffer, configs.env_name)

    logs = [{
        "step": 0,
        "reward": eval_policy(agent, env, configs.eval_episodes)[0]
    }]
    for t in trange(1, configs.max_timesteps + 1):
        log_info = agent.update(replay_buffer, model_buffer)
        # Save every 1e5 steps & last 5 checkpoints
        if (t % 100000 == 0) or (t >= int(9.8e5)
                                 and t % configs.eval_freq == 0):
            agent.save(f"{ckpt_dir}", t // configs.eval_freq)

        if (t > int(9.5e5) and
            (t % configs.eval_freq == 0)) or (t <= int(9.5e5) and t %
                                              (2 * configs.eval_freq) == 0):
            eval_reward, eval_time = eval_policy(agent, env,
                                                 configs.eval_episodes)
            log_info.update({
                "step": t,
                "reward": eval_reward,
                "eval_time": eval_time,
                "time": (time.time() - start_time) / 60
            })
            logs.append(log_info)
            logger.info(
                f"\n\n[#Step {t}] eval_reward: {eval_reward:.3f}, eval_time: {eval_time:.3f}, time: {log_info['time']:.3f}\n"

                # Actor loss info
                f"\treal_actor_loss: {log_info['real_actor_loss']:.3f}, max_real_actor_loss: {log_info['max_real_actor_loss']:.3}, min_real_actor_loss:{log_info['min_real_actor_loss']:.3f}\n"
                f"\tmodel_actor_loss: {log_info['model_actor_loss']:.3f}, max_model_actor_loss: {log_info['max_model_actor_loss']:.3}, min_model_actor_loss:{log_info['min_model_actor_loss']:.3f}\n"
                f"\treal_adv: {log_info['real_adv']:.3f}, max_real_adv: {log_info['max_real_adv']:.3f}, min_real_adv: {log_info['min_real_adv']:.3f}\n"
                f"\tmodel_adv: {log_info['model_adv']:.3f}, max_model_adv: {log_info['max_model_adv']:.3f}, min_model_adv: {log_info['min_model_adv']:.3f}\n"

                # Value loss info
                f"\treal_value_loss: {log_info['real_value_loss']:.3f}, max_real_value_loss: {log_info['max_real_value_loss']:.3}, min_real_value_loss:{log_info['min_real_value_loss']:.3f}\n"
                f"\tmodel_value_loss: {log_info['model_value_loss']:.3f}, max_model_value_loss: {log_info['max_model_value_loss']:.3}, min_model_value_loss:{log_info['min_model_value_loss']:.3f}\n"
                f"\treal_v: {log_info['real_v']:.3f}, max_real_v: {log_info['max_real_v']:.3f}, min_real_v: {log_info['min_real_v']:.3f}\n"
                f"\tmodel_v: {log_info['real_v']:.3f}, max_model_v: {log_info['max_model_v']:.3f}, min_model_v: {log_info['min_model_v']:.3f}\n"

                # Critic loss info
                f"\treal_q1: {log_info['real_q1']:.3f}, max_real_q1: {log_info['max_real_q1']:.3f}, min_real_q1: {log_info['min_real_q1']:.3f}\n"
                f"\tmodel_q1: {log_info['model_q1']:.3f}, max_model_q1: {log_info['max_model_q1']:.3f}, min_model_q1: {log_info['min_model_q1']:.3f}\n"
                f"\treal_critic_loss: {log_info['real_critic_loss']:.3f}, max_real_critic_loss: {log_info['max_real_critic_loss']:.3f}, min_real_critic_loss: {log_info['min_real_critic_loss']:.3f}\n"
                f"\tmodel_critic_loss: {log_info['model_critic_loss']:.3f}, max_model_critic_loss: {log_info['max_model_critic_loss']:.3f}, min_model_critic_loss: {log_info['min_model_critic_loss']:.3f}\n"

                # Batch information
                f"\treal_batch_rewards: {log_info['real_batch_rewards']:.2f}, real_batch_rewards_min: {log_info['real_batch_rewards_min']:.2f}, real_batch_rewards_max: {log_info['real_batch_rewards_max']:.2f}\n"
                f"\treal_batch_actions: {log_info['real_batch_actions']:.2f}, real_batch_observations: {log_info['real_batch_observations']:.2f}, "
                f"real_batch_discounts: {log_info['real_batch_discounts']:.2f}\n"
                f"\tmodel_batch_rewards: {log_info['model_batch_rewards']:.2f}, model_batch_rewards_min: {log_info['model_batch_rewards_min']:.2f}, model_batch_rewards_max: {log_info['model_batch_rewards_max']:.2f}\n"
                f"\tmodel_batch_actions: {log_info['model_batch_actions']:.2f}, model_batch_observations: {log_info['model_batch_observations']:.2f}, "
                f"model_batch_discounts: {log_info['model_batch_discounts']:.2f}\n"
                f"\tmodel_buffer_size: {log_info['model_buffer_size']:.0f}, "
                f"model_buffer_ptr: {log_info['model_buffer_ptr']:.0f}\n")

    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"{configs.log_dir}/{configs.env_name}/{configs.algo}/{exp_name}.csv")
    final_reward = log_df["reward"].iloc[-10:].mean()
    logger.info(f"\nAvg eval reward = {final_reward:.2f}\n")

import functools
import os
from typing import Any, Callable, Dict, List, Sequence, Tuple

import d4rl
import distrax
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import checkpoints, train_state

from static_fn import static_fns
from utils import Batch, ReplayBuffer, get_training_data, target_update


##################
# Util Functions #
##################
LOG_STD_MAX = 2.
LOG_STD_MIN = -5.


def init_fn(initializer: str, gain: float = jnp.sqrt(2)):
    if initializer == "orthogonal":
        return nn.initializers.orthogonal(gain)
    elif initializer == "glorot_uniform":
        return nn.initializers.glorot_uniform()
    elif initializer == "glorot_normal":
        return nn.initializers.glorot_normal()
    return nn.initializers.lecun_normal()


class MLP(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    init_fn: Callable = nn.initializers.glorot_uniform()
    activate_final: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.init_fn)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = nn.relu(x)
        return x


################
# Actor-Critic #
################
class Critic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"

    def setup(self):
        self.net = MLP(self.hidden_dims,
                       init_fn=init_fn(self.initializer),
                       activate_final=True)
        self.out_layer = nn.Dense(1,
                                  kernel_init=init_fn(self.initializer, 1.0))

    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)
        x = self.net(x)
        q = self.out_layer(x)
        return q.squeeze(-1)

    def encode(self, observations: jnp.ndarray,
               actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)
        x = self.net(x)
        return x


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"

    def setup(self):
        self.critic1 = Critic(self.hidden_dims, initializer=self.initializer)
        self.critic2 = Critic(self.hidden_dims, initializer=self.initializer)

    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        q1 = self.critic1(observations, actions)
        q2 = self.critic2(observations, actions)
        return q1, q2

    def Q1(self, observations: jnp.ndarray,
           actions: jnp.ndarray) -> jnp.ndarray:
        q1 = self.critic1(observations, actions)
        return q1


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"

    def setup(self):
        self.net = MLP(self.hidden_dims,
                       init_fn=init_fn(self.initializer),
                       activate_final=True)
        self.out_layer = nn.Dense(1,
                                  kernel_init=init_fn(self.initializer, 1.0))

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        x = self.net(observations)
        v = self.out_layer(x)
        return v.squeeze(-1)


class Actor(nn.Module):
    act_dim: int
    max_action: float = 1.0
    temperature: float = 3.0
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"

    def setup(self):
        self.net = MLP(self.hidden_dims,
                       init_fn=init_fn(self.initializer),
                       activate_final=True)
        self.mu_layer = nn.Dense(self.act_dim,
                                 kernel_init=init_fn(self.initializer))
        self.log_std = self.param('log_std', nn.initializers.zeros,
                                  (self.act_dim, ))

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        x = self.net(observations)
        x = self.mu_layer(x)
        mean_action = nn.tanh(x) * self.max_action
        return mean_action

    def get_log_prob(self, observations: jnp.ndarray,
                     actions: jnp.ndarray) -> jnp.ndarray:
        x = self.net(observations)
        x = self.mu_layer(x)
        log_std = jnp.clip(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)
        mean_action = nn.tanh(x) * self.max_action
        action_distribution = distrax.MultivariateNormalDiag(
            mean_action, std * self.temperature)
        log_prob = action_distribution.log_prob(actions)
        return log_prob


##########################
# Probabilistic Ensemble #
##########################
class EnsembleDense(nn.Module):
    ensemble_num: int
    features: int
    use_bias: bool = True
    dtype: Any = jnp.float32
    precision: Any = None
    kernel_init: Callable = jax.nn.initializers.lecun_normal()
    bias_init: Callable = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs: jnp.array) -> jnp.array:
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = self.param(
            "kernel", self.kernel_init,
            (self.ensemble_num, inputs.shape[-1], self.features))
        kernel = jnp.asarray(kernel, self.dtype)
        y = jnp.einsum("ij,ijk->ik", inputs, kernel)
        if self.use_bias:
            bias = self.param("bias", self.bias_init,
                              (self.ensemble_num, self.features))
            bias = jnp.asarray(bias, self.dtype)
            y += bias
        return y


class GaussianMLP(nn.Module):
    ensemble_num: int
    out_dim: int
    hid_dim: int = 200
    max_log_var: float = 0.5
    min_log_var: float = -10.0

    def setup(self):
        self.l1 = EnsembleDense(ensemble_num=self.ensemble_num,
                                features=self.hid_dim,
                                name="fc1")
        self.l2 = EnsembleDense(ensemble_num=self.ensemble_num,
                                features=self.hid_dim,
                                name="fc2")
        self.l3 = EnsembleDense(ensemble_num=self.ensemble_num,
                                features=self.hid_dim,
                                name="fc3")
        self.mean_and_logvar = EnsembleDense(ensemble_num=self.ensemble_num,
                                             features=self.out_dim * 2,
                                             name="output")

    def __call__(self, x):
        x = nn.leaky_relu(self.l1(x))
        x = nn.leaky_relu(self.l2(x))
        x = nn.leaky_relu(self.l3(x))
        x = self.mean_and_logvar(x)

        mu, log_var = jnp.split(x, 2, axis=-1)
        log_var = self.max_log_var - jax.nn.softplus(self.max_log_var -
                                                     log_var)
        log_var = self.min_log_var + jax.nn.softplus(log_var -
                                                     self.min_log_var)
        return mu, log_var


###############
# Simple MIQL #
###############
class DynamicsModel:
    def __init__(self,
                 env_name: str = "hopper-medium-v2",
                 seed: int = 42,
                 ensemble_num: int = 7,
                 elite_num: int = 5,
                 holdout_num: int = 1000,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-5,
                 epochs: int = 200,
                 batch_size: int = 2048,
                 max_patience: int = 10,
                 model_dir: str = "./saved_dynamics_models"):

        # Model parameters
        self.seed = seed
        self.lr = lr
        self.static_fn = static_fns[env_name.split('-')[0].lower()]
        print(f'Load static_fn: {self.static_fn}')
        self.weight_decay = weight_decay
        self.ensemble_num = ensemble_num
        self.elite_num = elite_num
        self.elite_models = None
        self.holdout_num = holdout_num
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_patience = max_patience

        # Environment & ReplayBuffer
        print(f'Loading data for {env_name}')
        self.env = gym.make(env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim)
        self.replay_buffer.convert_D4RL(d4rl.qlearning_dataset(self.env))

        # Initilaize saving settings
        self.save_file = f"{model_dir}/{env_name}"
        self.save_dir = f"dynamics_models/{env_name}"
        self.elite_mask = np.eye(self.ensemble_num)[range(elite_num), :]

        # Initilaize the ensemble model
        rng = jax.random.PRNGKey(seed + 10)
        _, model_key = jax.random.split(rng, 2)
        self.model = GaussianMLP(ensemble_num=ensemble_num,
                                 out_dim=self.obs_dim + 1)
        dummy_model_inputs = jnp.ones(
            [ensemble_num, self.obs_dim + self.act_dim], dtype=jnp.float32)
        model_params = self.model.init(model_key, dummy_model_inputs)["params"]

        self.model_state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=model_params,
            tx=optax.adamw(learning_rate=self.lr,
                           weight_decay=self.weight_decay))

        # Normalize inputs
        self.obs_mean = None
        self.obs_std = None

    def load(self, filename):
        step = max([
            int(i.split("_")[-1]) for i in os.listdir(filename)
            if "dynamics_model_" in i
        ])
        self.model_state = checkpoints.restore_checkpoint(
            ckpt_dir=filename,
            target=self.model_state,
            step=step,
            prefix="dynamics_model_")
        elite_idx = np.loadtxt(f"{filename}/elite_models.txt",
                               dtype=np.int32)[:self.elite_num]
        self.elite_mask = np.eye(self.ensemble_num)[elite_idx, :]
        normalize_stat = np.load(f'{filename}/normalize_stat.npz')
        self.obs_mean = normalize_stat['obs_mean']
        self.obs_std = normalize_stat['obs_std']

    def train(self):
        (inputs, targets, holdout_inputs, holdout_targets, self.obs_mean,
         self.obs_std) = get_training_data(self.replay_buffer,
                                           self.ensemble_num, 0.01)
        patience, optimal_state, min_val_loss = 0, None, np.inf
        batch_num = int(np.ceil(len(inputs) / self.batch_size))
        res = []
        print(f'batch_num     = {batch_num}')
        print(f'inputs.shape  = {inputs.shape}')
        print(f'targets.shape = {targets.shape}')
        print(f'holdout_inputs.shape  = {holdout_inputs.shape}')
        print(f'holdout_targets.shape = {holdout_targets.shape}')

        # Loss functions
        @jax.jit
        def train_step(model_state: train_state.TrainState,
                       batch_inputs: jnp.ndarray, batch_targets: jnp.ndarray):
            def loss_fn(params, x, y):
                mu, log_var = self.model.apply({"params": params},
                                               x)  # (7, 14) ==> (7, 12)
                inv_var = jnp.exp(-log_var)  # (7, 12)
                mse_loss = jnp.square(mu - y)  # (7, 12)
                train_loss = jnp.mean(mse_loss * inv_var + log_var,
                                      axis=-1).sum()
                return train_loss, {
                    "mse_loss": mse_loss.mean(),
                    "var_loss": log_var.mean(),
                    "train_loss": train_loss
                }

            grad_fn = jax.vmap(jax.value_and_grad(loss_fn, has_aux=True),
                               in_axes=(None, 1, 1))
            (_, log_info), gradients = grad_fn(model_state.params,
                                               batch_inputs, batch_targets)
            log_info = jax.tree_map(functools.partial(jnp.mean, axis=0),
                                    log_info)
            gradients = jax.tree_map(functools.partial(jnp.mean, axis=0),
                                     gradients)
            new_model_state = model_state.apply_gradients(grads=gradients)
            return new_model_state, log_info

        @jax.jit
        def eval_step(model_state: train_state.TrainState,
                      batch_inputs: jnp.ndarray, batch_targets: jnp.ndarray):
            def loss_fn(params, x, y):
                mu, _ = self.model.apply({"params": params},
                                         x)  # (7, 14) ==> (7, 12)
                reward_loss = jnp.square(mu[:, -1] - y[:, -1]).mean()
                state_loss = jnp.square(mu[:, :-1] - y[:, :-1]).mean()
                mse_loss = jnp.mean(jnp.square(mu - y),
                                    axis=-1)  # (7, 12) ==> (7,)
                return mse_loss, {
                    "reward_loss": reward_loss,
                    "state_loss": state_loss,
                    "mse_loss": mse_loss
                }

            loss_fn = jax.vmap(loss_fn, in_axes=(None, 1, 1))
            loss, log_info = loss_fn(model_state.params, batch_inputs,
                                     batch_targets)
            log_info = jax.tree_map(functools.partial(jnp.mean, axis=0),
                                    log_info)
            return loss, log_info

        for epoch in range(self.epochs):
            shuffled_idxs = np.concatenate([
                np.random.permutation(np.arange(inputs.shape[0])).reshape(
                    1, -1) for _ in range(self.ensemble_num)
            ],
                                           axis=0)  # (7, 1000000)
            train_loss, mse_loss, var_loss = [], [], []
            for i in range(batch_num):
                batch_idxs = shuffled_idxs[:, i * self.batch_size:(i + 1) *
                                           self.batch_size]
                batch_inputs = inputs[batch_idxs]  # (7, 256, 14)
                batch_targets = targets[batch_idxs]  # (7, 256, 12)
                self.model_state, log_info = train_step(
                    self.model_state, batch_inputs, batch_targets)
                train_loss.append(log_info["train_loss"].item())
                mse_loss.append(log_info["mse_loss"].item())
                var_loss.append(log_info["var_loss"].item())

            val_loss, val_info = eval_step(self.model_state, holdout_inputs,
                                           holdout_targets)
            val_loss = jnp.mean(val_loss, axis=0)  # (N, 7) ==> (7,)
            mean_val_loss = jnp.mean(val_loss)
            if mean_val_loss < min_val_loss:
                optimal_state = self.model_state
                min_val_loss = mean_val_loss
                elite_models = jnp.argsort(val_loss)  # find elite models
                patience = 0
            else:
                patience += 1
            if epoch > 20 and patience > self.max_patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

            res.append(
                (epoch, sum(train_loss) / batch_num, sum(mse_loss) / batch_num,
                 sum(var_loss) / batch_num, mean_val_loss))
            print(f"Epoch #{epoch+1}: "
                  f"train_loss={sum(train_loss)/batch_num:.3f}\t"
                  f"mse_loss={sum(mse_loss)/batch_num:.3f}\t"
                  f"var_loss={sum(var_loss)/batch_num:.3f}\t"
                  f"val_loss={mean_val_loss:.3f}\t"
                  f"val_rew_loss={val_info['reward_loss']:.3f}\t"
                  f"val_state_loss={val_info['state_loss']:.3f}")

            if (epoch + 1) in [10, 50, 100, 150]:
                checkpoints.save_checkpoint(f"{self.save_dir}",
                                            optimal_state,
                                            step=epoch + 1,
                                            prefix="dynamics_model_",
                                            keep=10,
                                            overwrite=True)

        checkpoints.save_checkpoint(f"{self.save_dir}",
                                    optimal_state,
                                    step=epoch + 1,
                                    prefix="dynamics_model_",
                                    keep=10,
                                    overwrite=True)
        res_df = pd.DataFrame(res,
                              columns=[
                                  "epoch", "train_loss", "mse_loss",
                                  "var_loss", "val_loss"
                              ])
        res_df.to_csv(f"{self.save_dir}/train_log.csv")
        ckpt_loss, _ = eval_step(optimal_state, holdout_inputs,
                                 holdout_targets)
        ckpt_loss = jnp.mean(ckpt_loss, axis=0)
        with open(f"{self.save_dir}/elite_models.txt", "w") as f:
            for idx in elite_models:
                f.write(f"{idx}\n")
        elite_idx = elite_models.to_py()[:self.elite_num]
        self.elite_mask = np.eye(self.ensemble_num)[elite_idx, :]
        np.savez(f"{self.save_dir}/normalize_stat",
                 obs_mean=self.obs_mean,
                 obs_std=self.obs_std)

    def rollout(self, params, observations, actions, model_masks):
        @jax.jit
        def rollout_fn(observation, action, model_mask):
            x = jnp.concatenate([observation, action], axis=-1).reshape(1, -1)
            model_mu, _ = self.model.apply({"params": params}, x)
            observation_mu, reward_mu = jnp.split(model_mu, [self.obs_dim],
                                                  axis=-1)
            model_next_observation = observation + jnp.sum(
                model_mask * observation_mu, axis=0)
            model_reward = jnp.sum(model_mask * reward_mu, axis=0)
            return model_next_observation, model_reward

        next_observations, rewards = jax.vmap(rollout_fn,
                                              in_axes=(0, 0, 0))(observations,
                                                                 actions,
                                                                 model_masks)
        next_observations = self.denormalize(next_observations)
        return next_observations, rewards

    def step(self, key, observations, actions):
        model_idx = jax.random.randint(key,
                                       shape=(actions.shape[0], ),
                                       minval=0,
                                       maxval=self.elite_num)
        model_masks = self.elite_mask[model_idx].reshape(
            -1, self.ensemble_num, 1)
        next_observations, rewards = self.rollout(self.model_state.params,
                                                  observations, actions,
                                                  model_masks)
        terminals = self.static_fn.termination_fn(observations, actions,
                                                  next_observations)
        return next_observations, rewards.squeeze(), terminals.squeeze()

    def normalize(self, observations):
        new_observations = (observations - self.obs_mean) / self.obs_std
        return new_observations

    def denormalize(self, observations):
        new_observations = observations * self.obs_std + self.obs_mean
        return new_observations


class MIQLAgent:
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 max_action: float,
                 env_name: str,
                 hidden_dims: Sequence[int],
                 seed: int,
                 lr: float,
                 tau: float,
                 gamma: float,
                 expectile: float,
                 temperature: float,
                 max_timesteps: int,
                 horizon: int = 5,
                 action_noise: float = 0.2,
                 warmup_timesteps: float = 300000,
                 uncertainty: int = 1,
                 eta: float = 0.8,
                 real_ratio: float = 0.5,
                 quantile: float = 0.5,
                 batchs: str = "real_concat_concat",
                 initializer: str = "orthogonal"):

        self.act_dim = act_dim
        self.max_action = max_action
        self.expectile = expectile
        self.temperature = temperature
        self.gamma = gamma
        self.tau = tau
        self.action_noise = action_noise
        self.warmup_timesteps = warmup_timesteps
        self.batchs = batchs.split('_')

        self.rng = jax.random.PRNGKey(seed)
        actor_key, critic_key, value_key = jax.random.split(self.rng, 3)
        dummy_obs = jnp.ones([1, obs_dim], dtype=jnp.float32)
        dummy_act = jnp.ones([1, act_dim], dtype=jnp.float32)

        self.actor = Actor(act_dim=act_dim,
                           max_action=max_action,
                           temperature=temperature,
                           hidden_dims=hidden_dims,
                           initializer=initializer)
        actor_params = self.actor.init(actor_key, dummy_obs)["params"]
        schedule_fn = optax.cosine_decay_schedule(-lr, max_timesteps)
        self.actor_state = train_state.TrainState.create(
            apply_fn=Actor.apply,
            params=actor_params,
            tx=optax.chain(optax.scale_by_adam(),
                           optax.scale_by_schedule(schedule_fn)))

        self.critic = DoubleCritic(hidden_dims=hidden_dims, initializer=initializer)
        critic_params = self.critic.init(critic_key, dummy_obs, dummy_act)["params"]
        self.critic_target_params = critic_params
        self.critic_state = train_state.TrainState.create(
            apply_fn=DoubleCritic.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=lr))

        self.value = ValueCritic(hidden_dims, initializer=initializer)
        value_params = self.value.init(value_key, dummy_obs)["params"]
        self.value_state = train_state.TrainState.create(
            apply_fn=ValueCritic.apply,
            params=value_params,
            tx=optax.adam(learning_rate=lr))

        # Initialize the Dynamics Model
        self.model = DynamicsModel(env_name=env_name,
                                   seed=seed,
                                   ensemble_num=7,
                                   elite_num=5)
        self.horizon = horizon
        self.update_step = 0

        self.real_size = int(256 * real_ratio)
        self.uncertainty = uncertainty
        self.uncertainty_threshold = 0

    @functools.partial(jax.jit,
                       static_argnames=("self"),
                       device=jax.devices("cpu")[0])
    def sample_action(self, params: FrozenDict,
                      observation: jnp.ndarray) -> jnp.ndarray:
        sampled_action = self.actor.apply({"params": params}, observation)
        return sampled_action

    def value_train_step(self,
                         batch: Batch,
                         value_state: train_state.TrainState,
                         critic_target_params: FrozenDict) -> Tuple[Dict, train_state.TrainState]:
        q1, q2 = self.critic.apply({"params": critic_target_params}, batch.observations, batch.actions)
        qs = jnp.minimum(q1, q2)
        def loss_fn(params: FrozenDict, q: jnp.ndarray, observation: jnp.ndarray):
            v = self.value.apply({"params": params}, observation)
            weight = jnp.where(q-v>0, self.expectile, 1-self.expectile)
            value_loss = weight * jnp.square(q-v)
            return value_loss, {"value_loss": value_loss, "weight": weight, "v": v}
        grad_fn = jax.vmap(jax.value_and_grad(loss_fn, has_aux=True), in_axes=(None, 0, 0))
        (_, value_info), value_grads = grad_fn(value_state.params, qs, batch.observations)
        extra_log = {
            "real_value_loss": value_info["value_loss"][:self.real_size].mean(),
            "max_real_value_loss": value_info["value_loss"][:self.real_size].max(),
            "min_real_value_loss": value_info["value_loss"][:self.real_size].min(),
            "model_value_loss": value_info["value_loss"][self.real_size:].mean(),
            "max_model_value_loss": value_info["value_loss"][self.real_size:].max(),
            "min_model_value_loss": value_info["value_loss"][self.real_size:].min(),
            "real_v": value_info["v"][:self.real_size].mean(),
            "max_real_v": value_info["v"][:self.real_size].max(),
            "min_real_v": value_info["v"][:self.real_size].min(),
            "model_v": value_info["v"][self.real_size:].mean(),
            "max_model_v": value_info["v"][self.real_size:].max(),
            "min_model_v": value_info["v"][self.real_size:].min(),
            "real_weight": value_info["weight"][:self.real_size].mean(),
            "max_real_weight": value_info["weight"][:self.real_size].max(),
            "min_real_weight": value_info["weight"][:self.real_size].min(),
            "model_weight": value_info["weight"][self.real_size:].mean(),
            "max_model_weight": value_info["weight"][self.real_size:].max(),
            "min_model_weight": value_info["weight"][self.real_size:].min(),
        }
        value_info = jax.tree_map(functools.partial(jnp.mean, axis=0), value_info)
        value_grads = jax.tree_map(functools.partial(jnp.mean, axis=0), value_grads)
        value_info.update(extra_log)
        value_state = value_state.apply_gradients(grads=value_grads)
        return value_info, value_state

    def actor_train_step(self,
                         batch: Batch,
                         actor_state: train_state.TrainState,
                         value_params: FrozenDict,
                         critic_target_params: FrozenDict) -> Tuple[Dict, train_state.TrainState]:
        v = self.value.apply({"params": value_params}, batch.observations)
        q1, q2 = self.critic.apply({"params": critic_target_params}, batch.observations, batch.actions)
        q = jnp.minimum(q1, q2)
        exp_as = jnp.exp((q - v) * self.temperature)
        exp_as = jnp.minimum(exp_as, 100.0)
        def loss_fn(params, observation, action, exp_a):
            log_prob = self.actor.apply({"params": params}, observation, action, method=Actor.get_log_prob)
            actor_loss = -exp_a * log_prob
            return actor_loss, {"actor_loss": actor_loss, "log_prob": log_prob}
        grad_fn = jax.vmap(jax.value_and_grad(loss_fn, has_aux=True), in_axes=(None, 0, 0, 0))
        (_, actor_info), actor_grads = grad_fn(actor_state.params, batch.observations, batch.actions, exp_as)
        extra_log = {
            "real_actor_loss": actor_info["actor_loss"][:self.real_size].mean(),
            "max_real_actor_loss": actor_info["actor_loss"][:self.real_size].max(),
            "min_real_actor_loss": actor_info["actor_loss"][:self.real_size].min(),
            "model_actor_loss": actor_info["actor_loss"][self.real_size:].mean(),
            "max_model_actor_loss": actor_info["actor_loss"][self.real_size:].max(),
            "min_model_actor_loss": actor_info["actor_loss"][self.real_size:].min(),
            "real_log_prob": actor_info["log_prob"][:self.real_size].mean(),
            "max_real_log_prob": actor_info["log_prob"][:self.real_size].max(),
            "min_real_log_prob": actor_info["log_prob"][:self.real_size].min(),
            "model_log_prob": actor_info["log_prob"][self.real_size:].mean(),
            "max_model_log_prob": actor_info["log_prob"][self.real_size:].max(),
            "min_model_log_prob": actor_info["log_prob"][self.real_size:].min(),
            "real_adv": (q-v)[:self.real_size].mean(),
            "max_real_adv": (q-v)[:self.real_size].max(),
            "min_real_adv": (q-v)[:self.real_size].min(),
            "model_adv": (q-v)[self.real_size:].mean(),
            "max_model_adv": (q-v)[self.real_size:].max(),
            "min_model_adv": (q-v)[self.real_size:].min(),
            "real_v_actor": v[:self.real_size].mean(),
            "max_real_v_actor": v[:self.real_size].max(),
            "min_real_v_actor": v[:self.real_size].min(),
            "model_v_actor": v[self.real_size:].mean(),
            "max_model_v_actor": v[self.real_size:].max(),
            "min_model_v_actor": v[self.real_size:].min(),
        }
        actor_info = jax.tree_map(functools.partial(jnp.mean, axis=0), actor_info)
        actor_grads = jax.tree_map(functools.partial(jnp.mean, axis=0), actor_grads)
        actor_info.update(extra_log)
        actor_state = actor_state.apply_gradients(grads=actor_grads)
        return actor_info, actor_state

    def critic_train_step(self,
                          batch: Batch,
                          critic_state: train_state.TrainState,
                          value_params: FrozenDict) -> Tuple[Dict, train_state.TrainState]:
        next_v = self.value.apply({"params": value_params}, batch.next_observations)
        target_qs = batch.rewards + self.gamma * batch.discounts * next_v
        def loss_fn(params: FrozenDict, observation: jnp.ndarray, action: jnp.ndarray, target_q: jnp.ndarray):
            q1, q2 = self.critic.apply({"params": params}, observation, action)
            critic_loss = (q1 - target_q)**2 + (q2 - target_q)**2
            return critic_loss, {"critic_loss": critic_loss, "q1": q1, "q2": q2}
        grad_fn = jax.vmap(jax.value_and_grad(loss_fn, has_aux=True), in_axes=(None, 0, 0, 0))
        (_, critic_info), critic_grads = grad_fn(critic_state.params, batch.observations, batch.actions, target_qs)
        extra_log = {
            "real_q1": critic_info["q1"][:self.real_size].mean(),
            "max_real_q1": critic_info["q1"][:self.real_size].max(),
            "min_real_q1": critic_info["q1"][:self.real_size].min(),
            "model_q1": critic_info["q1"][self.real_size:].mean(),
            "max_model_q1": critic_info["q1"][self.real_size:].max(),
            "min_model_q1": critic_info["q1"][self.real_size:].min(),
            "real_q2": critic_info["q2"][:self.real_size].mean(),
            "max_real_q2": critic_info["q2"][:self.real_size].max(),
            "min_real_q2": critic_info["q2"][:self.real_size].min(),
            "model_q2": critic_info["q2"][self.real_size:].mean(),
            "max_model_q2": critic_info["q2"][self.real_size:].max(),
            "min_model_q2": critic_info["q2"][self.real_size:].min(),
            "real_target_q": target_qs[:self.real_size].mean(),
            "max_real_target_q": target_qs[:self.real_size].max(),
            "min_real_target_q": target_qs[:self.real_size].min(),
            "model_target_q": target_qs[self.real_size:].mean(),
            "max_model_target_q": target_qs[self.real_size:].max(),
            "min_model_target_q": target_qs[self.real_size:].min(),
            "real_critic_loss": critic_info["critic_loss"][:self.real_size].mean(),
            "max_real_critic_loss": critic_info["critic_loss"][:self.real_size].max(),
            "min_real_critic_loss": critic_info["critic_loss"][:self.real_size].min(),
            "model_critic_loss": critic_info["critic_loss"][self.real_size:].mean(),
            "max_model_critic_loss": critic_info["critic_loss"][self.real_size:].max(),
            "min_model_critic_loss": critic_info["critic_loss"][self.real_size:].min(),
        }
        critic_info = jax.tree_map(functools.partial(jnp.mean, axis=0), critic_info)
        critic_grads = jax.tree_map(functools.partial(jnp.mean, axis=0), critic_grads)
        critic_state = critic_state.apply_gradients(grads=critic_grads)
        critic_info.update(extra_log)
        return critic_info, critic_state

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self,
                   batchs: List,
                   actor_state: train_state.TrainState,
                   value_state: train_state.TrainState,
                   critic_state: train_state.TrainState,
                   critic_target_params: FrozenDict):
        value_info, new_value_state = self.value_train_step(
            batchs[0], value_state, critic_target_params)
        actor_info, new_actor_state = self.actor_train_step(
            batchs[1], actor_state, new_value_state.params,
            critic_target_params)
        critic_info, new_critic_state = self.critic_train_step(
            batchs[2], critic_state, new_value_state.params)
        new_critic_target_params = target_update(new_critic_state.params,
                                                 critic_target_params,
                                                 self.tau)
        return new_actor_state, new_value_state, new_critic_state, new_critic_target_params, {
            **actor_info, **value_info, **critic_info}

    def update(self, replay_buffer, model_buffer):
        if (self.update_step >= self.warmup_timesteps) and (self.update_step % 1000 == 0):
            observations = replay_buffer.sample(10000).observations
            for _ in range(self.horizon):
                self.rng, rollout_key = jax.random.split(self.rng, 2)
                # sample action with action noises
                actions = (self.sample_action(self.actor_state.params, observations) + np.random.normal(
                    0, self.action_noise * self.max_action, size=(len(observations), self.act_dim))).clip(
                        -self.max_action + 1e-4, self.max_action - 1e-4)
                normalized_observations = self.model.normalize(observations)
                next_observations, rewards, dones = self.model.step(
                    rollout_key, normalized_observations, actions)
                nonterminal_mask = ~dones
                rewards = rewards / replay_buffer.reward_delta * 1000
                model_buffer.add_batch(observations, actions, next_observations, rewards, dones)
                if nonterminal_mask.sum() == 0:
                    print(f'[Breaking early]: {nonterminal_mask.shape}')
                    break
                observations = next_observations[nonterminal_mask]

        # sample from real & model buffer
        if (self.update_step >= self.warmup_timesteps):
            real_batch = replay_buffer.sample(256)
            model_batch = model_buffer.sample(256 - self.real_size)
            batch = Batch(
                observations=np.concatenate([real_batch.observations[:self.real_size], model_batch.observations], axis=0),
                actions = np.concatenate([real_batch.actions[:self.real_size], model_batch.actions], axis=0),
                rewards = np.concatenate([real_batch.rewards[:self.real_size], model_batch.rewards], axis=0),
                discounts = np.concatenate([real_batch.discounts[:self.real_size], model_batch.discounts], axis=0),
                next_observations = np.concatenate([real_batch.next_observations[:self.real_size], model_batch.next_observations], axis=0)
            )
        else:
            real_batch = replay_buffer.sample(256)
            batch = real_batch

        batch_dicts = {"real": real_batch, "concat": batch}
        update_batchs = [batch_dicts[i] for i in self.batchs]

        (self.actor_state, self.value_state, self.critic_state, self.critic_target_params,
         log_info) = self.train_step(update_batchs, self.actor_state, self.value_state, 
                                     self.critic_state, self.critic_target_params)

        log_info['real_batch_rewards'] = real_batch.rewards.sum() / 2
        log_info['real_batch_rewards_min'] = real_batch.rewards.min()
        log_info['real_batch_rewards_max'] = real_batch.rewards.max()
        log_info['real_batch_actions'] = abs(real_batch.actions).reshape(-1).sum() / 2
        log_info['real_batch_observations'] = abs(real_batch.observations).mean(0).sum()
        log_info['real_batch_discounts'] = real_batch.discounts.sum()

        if (self.update_step >= self.warmup_timesteps):
            log_info['model_batch_rewards'] = model_batch.rewards.sum()
            log_info['model_batch_rewards_min'] = model_batch.rewards.min()
            log_info['model_batch_rewards_max'] = model_batch.rewards.max()
            log_info['model_batch_actions'] = abs(model_batch.actions).reshape(-1).sum()
            log_info['model_batch_observations'] = abs(model_batch.observations).mean(0).sum()
            log_info['model_batch_discounts'] = model_batch.discounts.sum()
        else:
            log_info['model_batch_rewards'] = 0
            log_info['model_batch_rewards_min'] = 0
            log_info['model_batch_rewards_max'] = 0
            log_info['model_batch_actions'] = 0
            log_info['model_batch_observations'] = 0
            log_info['model_batch_discounts'] = 0
        log_info['model_buffer_size'] = model_buffer.size
        log_info['model_buffer_ptr'] = model_buffer.ptr
        self.update_step += 1
        return log_info

    def save(self, fname: str, cnt: int):
        checkpoints.save_checkpoint(fname,
                                    self.actor_state,
                                    cnt,
                                    prefix="actor_",
                                    keep=20,
                                    overwrite=True)
        checkpoints.save_checkpoint(fname,
                                    self.critic_state,
                                    cnt,
                                    prefix="critic_",
                                    keep=20,
                                    overwrite=True)
        checkpoints.save_checkpoint(fname,
                                    self.value_state,
                                    cnt,
                                    prefix="value_",
                                    keep=20,
                                    overwrite=True)


############
# CDA_MIQL #
############
MAX_UNCERTAINTY = 1.0
class DynamicsModel_CDA:
    def __init__(self,
                 env_name: str = "hopper-medium-v2",
                 seed: int = 42,
                 ensemble_num: int = 7,
                 elite_num: int = 5,
                 holdout_num: int = 1000,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-5,
                 epochs: int = 200,
                 batch_size: int = 256,
                 max_patience: int = 10,
                 model_dir: str = "./dynamics_models"):

        # Model parameters
        self.seed = seed
        self.lr = lr
        self.static_fn = static_fns[env_name.split('-')[0].lower()]
        print(f'Load static_fn: {self.static_fn}')
        self.weight_decay = weight_decay
        self.ensemble_num = ensemble_num
        self.elite_num = elite_num
        self.elite_models = None
        self.holdout_num = holdout_num
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_patience = max_patience

        # Environment & ReplayBuffer
        print(f'Loading data for {env_name}')
        self.env = gym.make(env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim)
        self.replay_buffer.convert_D4RL(d4rl.qlearning_dataset(self.env))

        # Initilaize saving settings
        self.save_file = f"{model_dir}/{env_name}"
        self.save_dir = f"{model_dir}/{env_name}"
        self.elite_mask = np.eye(self.ensemble_num)[range(elite_num), :]

        # Initilaize the ensemble model
        rng = jax.random.PRNGKey(seed+10)
        _, model_key = jax.random.split(rng, 2)
        self.model = GaussianMLP(ensemble_num=ensemble_num, out_dim=self.obs_dim+1)
        dummy_model_inputs = jnp.ones([ensemble_num, self.obs_dim+self.act_dim], dtype=jnp.float32)
        model_params = self.model.init(model_key, dummy_model_inputs)["params"]

        self.model_state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=model_params,
            tx=optax.adamw(learning_rate=self.lr, weight_decay=self.weight_decay))

        # Normalize inputs
        self.obs_mean = None
        self.obs_std = None 

    def load(self, filename):
        step = max([int(i.split("_")[-1]) for i in os.listdir(filename) if "dynamics_model_" in i])
        self.model_state = checkpoints.restore_checkpoint(ckpt_dir=filename,
                                                          target=self.model_state,
                                                          step=step,
                                                          prefix="dynamics_model_")
        elite_idx = np.loadtxt(f"{filename}/elite_models.txt", dtype=np.int32)[:self.elite_num]
        self.elite_mask = np.eye(self.ensemble_num)[elite_idx, :]
        normalize_stat = np.load(f'{filename}/normalize_stat.npz')
        self.obs_mean = normalize_stat['obs_mean']
        self.obs_std = normalize_stat['obs_std']

    def train(self):
        (inputs, targets, holdout_inputs, holdout_targets, self.obs_mean, self.obs_std) = get_training_data(self.replay_buffer, self.ensemble_num, 0.03)
        patience, optimal_state, min_val_loss = 0, None, np.inf
        batch_num = int(np.ceil(len(inputs) / self.batch_size))
        res = []
        print(f'batch_num     = {batch_num}')
        print(f'inputs.shape  = {inputs.shape}')
        print(f'targets.shape = {targets.shape}') 
        print(f'holdout_inputs.shape  = {holdout_inputs.shape}')
        print(f'holdout_targets.shape = {holdout_targets.shape}') 

        # Loss functions
        @jax.jit
        def train_step(model_state: train_state.TrainState, batch_inputs: jnp.ndarray, batch_targets: jnp.ndarray):
            def loss_fn(params, x, y):
                mu, log_var = self.model.apply({"params": params}, x)  # (7, 14) ==> (7, 12)
                inv_var = jnp.exp(-log_var)    # (7, 12)
                mse_loss = jnp.square(mu - y)  # (7, 12)
                obs_mse_loss, rew_mse_loss = jnp.split(mse_loss, [self.obs_dim], axis=-1)
                obs_inv_var, rew_inv_var = jnp.split(inv_var, [self.obs_dim], axis=-1)
                obs_log_var, rew_log_var = jnp.split(log_var, [self.obs_dim], axis=-1)
                obs_loss = jnp.mean(obs_mse_loss * obs_inv_var + obs_log_var, axis=-1).sum()
                rew_loss = jnp.mean(rew_mse_loss * rew_inv_var + rew_log_var, axis=-1).sum()
                train_loss = obs_loss + rew_loss
                # train_loss = jnp.mean(mse_loss * inv_var + log_var, axis=-1).sum() 
                return train_loss, {"mse_loss": mse_loss.mean(), "var_loss": log_var.mean(), "train_loss": train_loss}
            grad_fn = jax.vmap(jax.value_and_grad(loss_fn, has_aux=True), in_axes=(None, 1, 1))
            (_, log_info), gradients = grad_fn(model_state.params, batch_inputs, batch_targets)
            log_info = jax.tree_map(functools.partial(jnp.mean, axis=0), log_info)
            gradients = jax.tree_map(functools.partial(jnp.mean, axis=0), gradients)
            new_model_state = model_state.apply_gradients(grads=gradients)
            return new_model_state, log_info

        @jax.jit
        def eval_step(model_state: train_state.TrainState, batch_inputs: jnp.ndarray, batch_targets: jnp.ndarray):
            def loss_fn(params, x, y):
                mu, _ = self.model.apply({"params": params}, x)  # (7, 14) ==> (7, 12)
                reward_loss = jnp.square(mu[:, -1] - y[:, -1]).mean()
                state_loss = jnp.square(mu[:, :-1] - y[:, :-1]).mean()
                mse_loss = jnp.mean(jnp.square(mu - y), axis=-1)  # (7, 12) ==> (7,)
                return mse_loss, {"reward_loss": reward_loss, "state_loss": state_loss, "mse_loss": mse_loss}
            loss_fn = jax.vmap(loss_fn, in_axes=(None, 1, 1))
            loss, log_info = loss_fn(model_state.params, batch_inputs, batch_targets)
            log_info = jax.tree_map(functools.partial(jnp.mean, axis=0), log_info)
            return loss, log_info

        for epoch in range(self.epochs):
            shuffled_idxs = np.concatenate([np.random.permutation(np.arange(
                inputs.shape[0])).reshape(1, -1) for _ in range(self.ensemble_num)], axis=0)  # (7, 1000000)
            train_loss, mse_loss, var_loss = [], [], []
            for i in range(batch_num):
                batch_idxs = shuffled_idxs[:, i*self.batch_size:(i+1)*self.batch_size]
                batch_inputs = inputs[batch_idxs]    # (7, 256, 14)
                batch_targets = targets[batch_idxs]  # (7, 256, 12)
                self.model_state, log_info = train_step(self.model_state, batch_inputs, batch_targets)
                train_loss.append(log_info["train_loss"].item())
                mse_loss.append(log_info["mse_loss"].item())
                var_loss.append(log_info["var_loss"].item())

            val_loss, val_info = eval_step(self.model_state, holdout_inputs, holdout_targets)
            val_loss = jnp.mean(val_loss, axis=0)  # (N, 7) ==> (7,)
            mean_val_loss = jnp.mean(val_loss)
            if mean_val_loss < min_val_loss:
                optimal_state = self.model_state
                min_val_loss = mean_val_loss
                elite_models = jnp.argsort(val_loss)  # find elite models
                patience = 0
            else:
                patience += 1
            if epoch > 50 and patience > self.max_patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

            res.append((epoch, sum(train_loss)/batch_num, sum(mse_loss)/batch_num, sum(var_loss)/batch_num, mean_val_loss))
            print(f"Epoch #{epoch+1}: "
                  f"train_loss={sum(train_loss)/batch_num:.3f}\t"
                  f"mse_loss={sum(mse_loss)/batch_num:.3f}\t"
                  f"var_loss={sum(var_loss)/batch_num:.3f}\t"
                  f"val_loss={mean_val_loss:.3f}\t"
                  f"val_rew_loss={val_info['reward_loss']:.3f}\t"
                  f"val_state_loss={val_info['state_loss']:.3f}")

            if (epoch+1) in [5, 10, 20, 50]:
                checkpoints.save_checkpoint(f"{self.save_dir}", optimal_state, step=epoch+1, prefix="dynamics_model_", keep=10, overwrite=True)

        checkpoints.save_checkpoint(f"{self.save_dir}", optimal_state, step=epoch+1, prefix="dynamics_model_", keep=10, overwrite=True)
        res_df = pd.DataFrame(res, columns=["epoch", "train_loss", "mse_loss", "var_loss", "val_loss"])
        res_df.to_csv(f"{self.save_dir}/train_log.csv")
        ckpt_loss, _ = eval_step(optimal_state, holdout_inputs, holdout_targets)
        ckpt_loss = jnp.mean(ckpt_loss, axis=0)
        with open(f"{self.save_dir}/elite_models.txt", "w") as f:
            for idx in elite_models:
                f.write(f"{idx}\n")
        elite_idx = elite_models.to_py()[:self.elite_num]
        self.elite_mask = np.eye(self.ensemble_num)[elite_idx, :]
        np.savez(f"{self.save_dir}/normalize_stat", obs_mean=self.obs_mean, obs_std=self.obs_std)

    def rollout(self, params, observations, actions, model_masks):
        @jax.jit
        def rollout_fn(observation, action, model_mask):
            x = jnp.concatenate([observation, action], axis=-1).reshape(1, -1)
            model_mu, model_log_var = self.model.apply({"params": params}, x)
            observation_mu, reward_mu = jnp.split(model_mu, [self.obs_dim], axis=-1)
            model_next_observation = observation + jnp.sum(model_mask * observation_mu, axis=0)
            model_reward = jnp.sum(model_mask * reward_mu, axis=0)
            observation_mu += observation
            uncertainty1 = jnp.exp(model_log_var*0.5).sum(-1).max() / (self.obs_dim+1)
            uncertainty2 = jnp.linalg.norm(observation_mu - observation_mu.mean(0), axis=-1).max() / self.obs_dim
            return model_next_observation, model_reward, jnp.array([uncertainty1, uncertainty2])
        next_observations, rewards, uncertainties = jax.vmap(rollout_fn, in_axes=(0, 0, 0))(observations, actions, model_masks)
        next_observations = self.denormalize(next_observations)
        return next_observations, rewards, uncertainties

    def step(self, key, observations, actions):
        model_idx = jax.random.randint(key, shape=(actions.shape[0],), minval=0, maxval=self.elite_num)
        model_masks = self.elite_mask[model_idx].reshape(-1, self.ensemble_num, 1)
        next_observations, rewards, uncertainties = self.rollout(self.model_state.params, observations, actions, model_masks)
        terminals = self.static_fn.termination_fn(observations, actions, next_observations)
        return next_observations, rewards.squeeze(), terminals.squeeze(), uncertainties*100.0

    def normalize(self, observations):
        new_observations = (observations - self.obs_mean) / self.obs_std
        return new_observations

    def denormalize(self, observations):
        new_observations = observations * self.obs_std + self.obs_mean
        return new_observations


# add uncertainty thresholding
class MIQLAgent_CDA:
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 max_action: float,
                 env_name: str,
                 hidden_dims: Sequence[int],
                 seed: int,
                 lr: float,
                 tau: float,
                 gamma: float,
                 expectile: float,
                 temperature: float,
                 max_timesteps: int,
                 horizon: int = 5,
                 action_noise: float = 0.2,
                 warmup_timesteps: float = 300000,
                 uncertainty: int = 1,
                 eta: float = 0.8,
                 real_ratio: float = 0.5,
                 quantile: float = 0.5,
                 batchs: str = "real_concat_concat",
                 initializer: str = "orthogonal"):

        self.act_dim = act_dim
        self.max_action = max_action
        self.expectile = expectile
        self.temperature = temperature
        self.gamma = gamma
        self.tau = tau
        self.eta = eta
        self.action_noise = action_noise
        self.warmup_timesteps = warmup_timesteps
        self.batchs = batchs.split('_')
        self.quantile = quantile

        self.rng = jax.random.PRNGKey(seed)
        actor_key, critic_key, value_key = jax.random.split(self.rng, 3)
        dummy_obs = jnp.ones([1, obs_dim], dtype=jnp.float32)
        dummy_act = jnp.ones([1, act_dim], dtype=jnp.float32)

        self.actor = Actor(act_dim=act_dim,
                           max_action=max_action,
                           temperature=temperature,
                           hidden_dims=hidden_dims,
                           initializer=initializer)
        actor_params = self.actor.init(actor_key, dummy_obs)["params"]
        schedule_fn = optax.cosine_decay_schedule(-lr, max_timesteps)
        self.actor_state = train_state.TrainState.create(
            apply_fn=Actor.apply,
            params=actor_params,
            tx=optax.chain(optax.scale_by_adam(),
                           optax.scale_by_schedule(schedule_fn)))

        self.critic = DoubleCritic(hidden_dims=hidden_dims, initializer=initializer)
        critic_params = self.critic.init(critic_key, dummy_obs, dummy_act)["params"]
        self.critic_target_params = critic_params
        self.critic_state = train_state.TrainState.create(
            apply_fn=DoubleCritic.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=lr))

        self.value = ValueCritic(hidden_dims, initializer=initializer)
        value_params = self.value.init(value_key, dummy_obs)["params"]
        self.value_state = train_state.TrainState.create(
            apply_fn=ValueCritic.apply,
            params=value_params,
            tx=optax.adam(learning_rate=lr))

        # Initialize the Dynamics Model
        self.model = DynamicsModel_CDA(env_name=env_name,
                                       seed=seed,
                                       ensemble_num=7,
                                       elite_num=5)
        self.horizon = horizon
        self.update_step = 0

        self.real_size = int(256 * real_ratio)
        self.uncertainty = uncertainty
        self.uncertainty_threshold = 0

    @functools.partial(jax.jit, static_argnames=("self"), device=jax.devices("cpu")[0])
    def sample_action(self, params: FrozenDict, observation: jnp.ndarray) -> jnp.ndarray:
        sampled_action = self.actor.apply({"params": params}, observation)
        return sampled_action

    def value_train_step(self,
                         batch: Batch,
                         value_state: train_state.TrainState,
                         critic_target_params: FrozenDict) -> Tuple[Dict, train_state.TrainState]:
        q1, q2 = self.critic.apply({"params": critic_target_params}, batch.observations, batch.actions)
        qs = jnp.minimum(q1, q2)
        def loss_fn(params: FrozenDict, q: jnp.ndarray, observation: jnp.ndarray):
            v = self.value.apply({"params": params}, observation)
            weight = jnp.where(q-v>0, self.expectile, 1-self.expectile)
            value_loss = weight * jnp.square(q-v)
            return value_loss, {"value_loss": value_loss, "weight": weight, "v": v}
        grad_fn = jax.vmap(jax.value_and_grad(loss_fn, has_aux=True), in_axes=(None, 0, 0))
        (_, value_info), value_grads = grad_fn(value_state.params, qs, batch.observations)
        extra_log = {
            "real_value_loss": value_info["value_loss"][:self.real_size].mean(),
            "max_real_value_loss": value_info["value_loss"][:self.real_size].max(),
            "min_real_value_loss": value_info["value_loss"][:self.real_size].min(),
            "model_value_loss": value_info["value_loss"][self.real_size:].mean(),
            "max_model_value_loss": value_info["value_loss"][self.real_size:].max(),
            "min_model_value_loss": value_info["value_loss"][self.real_size:].min(),
            "real_v": value_info["v"][:self.real_size].mean(),
            "max_real_v": value_info["v"][:self.real_size].max(),
            "min_real_v": value_info["v"][:self.real_size].min(),
            "model_v": value_info["v"][self.real_size:].mean(),
            "max_model_v": value_info["v"][self.real_size:].max(),
            "min_model_v": value_info["v"][self.real_size:].min(),
            "real_weight": value_info["weight"][:self.real_size].mean(),
            "max_real_weight": value_info["weight"][:self.real_size].max(),
            "min_real_weight": value_info["weight"][:self.real_size].min(),
            "model_weight": value_info["weight"][self.real_size:].mean(),
            "max_model_weight": value_info["weight"][self.real_size:].max(),
            "min_model_weight": value_info["weight"][self.real_size:].min(),
        }
        value_info = jax.tree_map(functools.partial(jnp.mean, axis=0), value_info)
        value_grads = jax.tree_map(functools.partial(jnp.mean, axis=0), value_grads)
        value_info.update(extra_log)
        value_state = value_state.apply_gradients(grads=value_grads)
        return value_info, value_state

    def actor_train_step(self,
                         batch: Batch,
                         actor_state: train_state.TrainState,
                         value_params: FrozenDict,
                         critic_target_params: FrozenDict) -> Tuple[Dict, train_state.TrainState]:
        v = self.value.apply({"params": value_params}, batch.observations)
        q1, q2 = self.critic.apply({"params": critic_target_params}, batch.observations, batch.actions)
        q = jnp.minimum(q1, q2)
        exp_as = jnp.exp((q - v) * self.temperature)
        exp_as = jnp.minimum(exp_as, 100.0)
        def loss_fn(params, observation, action, exp_a):
            log_prob = self.actor.apply({"params": params}, observation, action, method=Actor.get_log_prob)
            actor_loss = -exp_a * log_prob
            return actor_loss, {"actor_loss": actor_loss, "log_prob": log_prob}
        grad_fn = jax.vmap(jax.value_and_grad(loss_fn, has_aux=True), in_axes=(None, 0, 0, 0))
        (_, actor_info), actor_grads = grad_fn(actor_state.params, batch.observations, batch.actions, exp_as)
        extra_log = {
            "real_actor_loss": actor_info["actor_loss"][:self.real_size].mean(),
            "max_real_actor_loss": actor_info["actor_loss"][:self.real_size].max(),
            "min_real_actor_loss": actor_info["actor_loss"][:self.real_size].min(),
            "model_actor_loss": actor_info["actor_loss"][self.real_size:].mean(),
            "max_model_actor_loss": actor_info["actor_loss"][self.real_size:].max(),
            "min_model_actor_loss": actor_info["actor_loss"][self.real_size:].min(),
            "real_log_prob": actor_info["log_prob"][:self.real_size].mean(),
            "max_real_log_prob": actor_info["log_prob"][:self.real_size].max(),
            "min_real_log_prob": actor_info["log_prob"][:self.real_size].min(),
            "model_log_prob": actor_info["log_prob"][self.real_size:].mean(),
            "max_model_log_prob": actor_info["log_prob"][self.real_size:].max(),
            "min_model_log_prob": actor_info["log_prob"][self.real_size:].min(),
            "real_adv": (q-v)[:self.real_size].mean(),
            "max_real_adv": (q-v)[:self.real_size].max(),
            "min_real_adv": (q-v)[:self.real_size].min(),
            "model_adv": (q-v)[self.real_size:].mean(),
            "max_model_adv": (q-v)[self.real_size:].max(),
            "min_model_adv": (q-v)[self.real_size:].min(),
            "real_v_actor": v[:self.real_size].mean(),
            "max_real_v_actor": v[:self.real_size].max(),
            "min_real_v_actor": v[:self.real_size].min(),
            "model_v_actor": v[self.real_size:].mean(),
            "max_model_v_actor": v[self.real_size:].max(),
            "min_model_v_actor": v[self.real_size:].min(),
        }
        actor_info = jax.tree_map(functools.partial(jnp.mean, axis=0), actor_info)
        actor_grads = jax.tree_map(functools.partial(jnp.mean, axis=0), actor_grads)
        actor_info.update(extra_log)
        actor_state = actor_state.apply_gradients(grads=actor_grads)
        return actor_info, actor_state

    def critic_train_step(self,
                          batch: Batch,
                          critic_state: train_state.TrainState,
                          value_params: FrozenDict) -> Tuple[Dict, train_state.TrainState]:
        next_v = self.value.apply({"params": value_params}, batch.next_observations)
        target_qs = batch.rewards + self.gamma * batch.discounts * next_v
        def loss_fn(params: FrozenDict, observation: jnp.ndarray, action: jnp.ndarray, target_q: jnp.ndarray):
            q1, q2 = self.critic.apply({"params": params}, observation, action)
            critic_loss = (q1 - target_q)**2 + (q2 - target_q)**2
            return critic_loss, {"critic_loss": critic_loss, "q1": q1, "q2": q2}
        grad_fn = jax.vmap(jax.value_and_grad(loss_fn, has_aux=True), in_axes=(None, 0, 0, 0))
        (_, critic_info), critic_grads = grad_fn(critic_state.params, batch.observations, batch.actions, target_qs)
        extra_log = {
            "real_q1": critic_info["q1"][:self.real_size].mean(),
            "max_real_q1": critic_info["q1"][:self.real_size].max(),
            "min_real_q1": critic_info["q1"][:self.real_size].min(),
            "model_q1": critic_info["q1"][self.real_size:].mean(),
            "max_model_q1": critic_info["q1"][self.real_size:].max(),
            "min_model_q1": critic_info["q1"][self.real_size:].min(),
            "real_q2": critic_info["q2"][:self.real_size].mean(),
            "max_real_q2": critic_info["q2"][:self.real_size].max(),
            "min_real_q2": critic_info["q2"][:self.real_size].min(),
            "model_q2": critic_info["q2"][self.real_size:].mean(),
            "max_model_q2": critic_info["q2"][self.real_size:].max(),
            "min_model_q2": critic_info["q2"][self.real_size:].min(),
            "real_target_q": target_qs[:self.real_size].mean(),
            "max_real_target_q": target_qs[:self.real_size].max(),
            "min_real_target_q": target_qs[:self.real_size].min(),
            "model_target_q": target_qs[self.real_size:].mean(),
            "max_model_target_q": target_qs[self.real_size:].max(),
            "min_model_target_q": target_qs[self.real_size:].min(),
            "real_critic_loss": critic_info["critic_loss"][:self.real_size].mean(),
            "max_real_critic_loss": critic_info["critic_loss"][:self.real_size].max(),
            "min_real_critic_loss": critic_info["critic_loss"][:self.real_size].min(),
            "model_critic_loss": critic_info["critic_loss"][self.real_size:].mean(),
            "max_model_critic_loss": critic_info["critic_loss"][self.real_size:].max(),
            "min_model_critic_loss": critic_info["critic_loss"][self.real_size:].min(),
        }
        critic_info = jax.tree_map(functools.partial(jnp.mean, axis=0), critic_info)
        critic_grads = jax.tree_map(functools.partial(jnp.mean, axis=0), critic_grads)
        critic_state = critic_state.apply_gradients(grads=critic_grads)
        critic_info.update(extra_log)
        return critic_info, critic_state

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self,
                   batchs: List,
                   actor_state: train_state.TrainState,
                   value_state: train_state.TrainState,
                   critic_state: train_state.TrainState,
                   critic_target_params: FrozenDict):
        value_info, new_value_state = self.value_train_step(
            batchs[0], value_state, critic_target_params)
        actor_info, new_actor_state = self.actor_train_step(
            batchs[1], actor_state, new_value_state.params,
            critic_target_params)
        critic_info, new_critic_state = self.critic_train_step(
            batchs[2], critic_state, new_value_state.params)
        new_critic_target_params = target_update(new_critic_state.params,
                                                 critic_target_params,
                                                 self.tau)
        return new_actor_state, new_value_state, new_critic_state, new_critic_target_params, {
            **actor_info, **value_info, **critic_info}

    def update(self, replay_buffer, model_buffer):
        if (self.update_step >= self.warmup_timesteps) and (self.update_step % 1000 == 0):
            observations = replay_buffer.sample(10000).observations
            for _ in range(self.horizon):
                self.rng, rollout_key = jax.random.split(self.rng, 2)
                actions = (self.sample_action(self.actor_state.params, observations) +
                    np.random.normal(0, self.action_noise, size=(len(observations), self.act_dim))).clip(-self.max_action, self.max_action)
                normalized_observations = self.model.normalize(observations)
                next_observations, rewards, dones, uncertainties = self.model.step(rollout_key, normalized_observations, actions)
                if self.uncertainty_threshold == 0:
                    self.uncertainty_threshold = np.quantile(uncertainties[:, self.uncertainty], self.quantile)
                else:
                    self.uncertainty_threshold = self.uncertainty_threshold*self.eta + (1-self.eta)*np.quantile(
                        uncertainties[:, self.uncertainty], self.quantile)
                nonterminal_mask = ~dones
                uncertainty_mask = uncertainties[:, self.uncertainty] < self.uncertainty_threshold
                mask = nonterminal_mask & uncertainty_mask
                rewards = rewards / replay_buffer.reward_delta * 1000
                model_buffer.add_batch(observations[uncertainty_mask], actions[uncertainty_mask], next_observations[uncertainty_mask],
                                       rewards[uncertainty_mask], dones[uncertainty_mask], uncertainties[uncertainty_mask])
                if mask.sum() == 0: break
                observations = next_observations[mask]

        if (self.update_step >= self.warmup_timesteps):
            real_batch = replay_buffer.sample(256)
            model_batch = model_buffer.sample(256 - self.real_size)
            batch = Batch(
                observations=np.concatenate([real_batch.observations[:self.real_size], model_batch.observations], axis=0),
                actions = np.concatenate([real_batch.actions[:self.real_size], model_batch.actions], axis=0),
                rewards = np.concatenate([real_batch.rewards[:self.real_size], model_batch.rewards], axis=0),
                discounts = np.concatenate([real_batch.discounts[:self.real_size], model_batch.discounts], axis=0),
                next_observations = np.concatenate([real_batch.next_observations[:self.real_size], model_batch.next_observations], axis=0)
            )
        else:
            real_batch = replay_buffer.sample(256)
            batch = real_batch

        batch_dicts = {"real": real_batch, "concat": batch}
        update_batchs = [batch_dicts[i] for i in self.batchs]

        (self.actor_state, self.value_state, self.critic_state, self.critic_target_params,
         log_info) = self.train_step(update_batchs, self.actor_state, self.value_state, 
                                     self.critic_state, self.critic_target_params)

        log_info['real_batch_rewards'] = real_batch.rewards.mean()
        log_info['real_batch_rewards_min'] = real_batch.rewards.min()
        log_info['real_batch_rewards_max'] = real_batch.rewards.max()
        log_info['real_batch_actions'] = abs(real_batch.actions).sum(-1).mean()
        log_info['real_batch_observations'] = abs(real_batch.observations).sum(-1).mean()
        log_info['real_batch_discounts'] = real_batch.discounts.mean()

        if (self.update_step >= self.warmup_timesteps):
            log_info['model_uncertainty1s_min'] = model_batch.stds[:, 0].min()
            log_info['model_uncertainty1s_max'] = model_batch.stds[:, 0].max()
            log_info['model_uncertainty1s_med'] = np.quantile(model_batch.stds[:, 0], 0.5)
            log_info['model_uncertainty2s_min'] = model_batch.stds[:, 1].min()
            log_info['model_uncertainty2s_max'] = model_batch.stds[:, 1].max()
            log_info['model_uncertainty2s_med'] = np.quantile(model_batch.stds[:, 1], 0.5)
            
            log_info['model_batch_rewards'] = model_batch.rewards.mean()
            log_info['model_batch_rewards_min'] = model_batch.rewards.min()
            log_info['model_batch_rewards_max'] = model_batch.rewards.max()
            log_info['model_batch_actions'] = abs(model_batch.actions).sum(-1).mean()
            log_info['model_batch_observations'] = abs(model_batch.observations).sum(-1).mean()
            log_info['model_batch_discounts'] = model_batch.discounts.mean()
        else:
            log_info['model_uncertainty1s_min'] = 0
            log_info['model_uncertainty1s_max'] = 0
            log_info['model_uncertainty1s_med'] = 0
            log_info['model_uncertainty2s_min'] = 0
            log_info['model_uncertainty2s_max'] = 0
            log_info['model_uncertainty2s_med'] = 0

            log_info['model_batch_rewards'] = 0
            log_info['model_batch_rewards_min'] = 0
            log_info['model_batch_rewards_max'] = 0
            log_info['model_batch_actions'] = 0
            log_info['model_batch_observations'] = 0
            log_info['model_batch_discounts'] = 0
        log_info['model_buffer_size'] = model_buffer.size
        log_info['model_buffer_ptr'] = model_buffer.ptr
        self.update_step += 1
        return log_info

    def save(self, fname: str, cnt: int):
        checkpoints.save_checkpoint(fname, self.actor_state, cnt, prefix="actor_", keep=20, overwrite=True)
        checkpoints.save_checkpoint(fname, self.critic_state, cnt, prefix="critic_", keep=20, overwrite=True)
        checkpoints.save_checkpoint(fname, self.value_state, cnt, prefix="value_", keep=20, overwrite=True)

from typing import Any, Callable, Sequence, Tuple
from flax import linen as nn
from flax import serialization
from flax.core import FrozenDict
from flax.training import train_state, checkpoints
import functools
import gym
import d4rl
import distrax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import optax
import os
from tqdm import trange
from static_fn import static_fns
from utils import Batch, ReplayBuffer, get_training_data, target_update


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

def atanh(x: jnp.ndarray):
    one_plus_x = jnp.clip(1 + x, a_min=1e-6)
    one_minus_x = jnp.clip(1 - x, a_min=1e-6)
    return 0.5 * jnp.log(one_plus_x / one_minus_x)

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


class Critic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256, 256)
    initializer: str = "orthogonal"

    def setup(self):
        self.net = MLP(self.hidden_dims, init_fn=init_fn(self.initializer), activate_final=True)
        self.out_layer = nn.Dense(1, kernel_init=init_fn(self.initializer, 1.0))

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)
        x = self.net(x)
        q = self.out_layer(x)
        return q.squeeze(-1)

    def encode(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)
        x = self.net(x)
        return x


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256, 256)
    initializer: str = "orthogonal"

    def setup(self):
        self.critic1 = Critic(self.hidden_dims, initializer=self.initializer)
        self.critic2 = Critic(self.hidden_dims, initializer=self.initializer)

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        q1 = self.critic1(observations, actions)
        q2 = self.critic2(observations, actions)
        return q1, q2

    def Q1(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        q1 = self.critic1(observations, actions)
        return q1


class Actor(nn.Module):
    act_dim: int
    max_action: float = 1.0
    hidden_dims: Sequence[int] = (256, 256, 256)
    initializer: str = "glorot_uniform"

    def setup(self):
        self.net = MLP(self.hidden_dims, init_fn=init_fn(self.initializer), activate_final=True)
        self.mu_layer = nn.Dense(self.act_dim, kernel_init=init_fn(self.initializer, 1e-2))
        self.std_layer = nn.Dense(self.act_dim, kernel_init=init_fn(self.initializer, 1e-2))

    def __call__(self, rng: Any, observation: jnp.ndarray) -> jnp.ndarray:
        x = self.net(observation)
        mu = self.mu_layer(x)
        log_std = self.std_layer(x)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)

        mean_action = nn.tanh(mu)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mu, std),
            distrax.Block(distrax.Tanh(), ndims=1))
        sampled_action, logp = action_distribution.sample_and_log_prob(seed=rng)
        return mean_action*self.max_action, sampled_action*self.max_action, logp

    def get_logp(self, observation: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = self.net(observation)
        mu = self.mu_layer(x)
        log_std = self.std_layer(x)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)
        action_distribution = distrax.Normal(mu, std)
        raw_action = atanh(action)
        logp = action_distribution.log_prob(raw_action).sum(-1)
        logp -= 2*(jnp.log(2) - raw_action - jax.nn.softplus(-2*raw_action)).sum(-1)
        return logp


class Scalar(nn.Module):
    init_value: float

    def setup(self):
        self.value = self.param("value", lambda x: self.init_value)
    
    def __call__(self):
        return self.value


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
        kernel = self.param("kernel", self.kernel_init,
                            (self.ensemble_num, inputs.shape[-1], self.features))
        kernel = jnp.asarray(kernel, self.dtype)
        y = jnp.einsum("ij,ijk->ik", inputs, kernel)
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.ensemble_num, self.features))
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
        self.l1 = EnsembleDense(ensemble_num=self.ensemble_num, features=self.hid_dim, name="fc1")
        self.l2 = EnsembleDense(ensemble_num=self.ensemble_num, features=self.hid_dim, name="fc2")
        self.l3 = EnsembleDense(ensemble_num=self.ensemble_num, features=self.hid_dim, name="fc3")
        self.mean_and_logvar = EnsembleDense(ensemble_num=self.ensemble_num, features=self.out_dim*2, name="output")

    def __call__(self, x):
        x = nn.leaky_relu(self.l1(x))
        x = nn.leaky_relu(self.l2(x))
        x = nn.leaky_relu(self.l3(x))
        x = self.mean_and_logvar(x)

        mu, log_var = jnp.split(x, 2, axis=-1)
        log_var = self.max_log_var - jax.nn.softplus(self.max_log_var - log_var)
        log_var = self.min_log_var + jax.nn.softplus(log_var - self.min_log_var)
        return mu, log_var


class DynamicsModel:
    def __init__(self,
                 env_name: str = "hopper-medium-v2",
                 seed: int = 42,
                 ensemble_num: int = 7,
                 elite_num: int = 5,
                 holdout_ratio: float = 0.01,
                 lr: float = 1e-3,
                 weight_decay: float = 5e-5,
                 epochs: int = 500,
                 batch_size: int = 256,
                 max_patience: int = 10,
                 model_dir: str = "./dynamics_models",
                 noise_scale: float = 0.0):

        # Model parameters
        self.seed = seed
        self.lr = lr
        self.static_fn = static_fns[env_name.split('-')[0].lower()]
        print(f'Load static_fn: {self.static_fn}')
        self.weight_decay = weight_decay
        self.ensemble_num = ensemble_num
        self.elite_num = elite_num
        self.elite_models = None
        self.holdout_ratio = holdout_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_patience = max_patience
        self.noise_scale = noise_scale

        # Environment & ReplayBuffer
        print(f'Loading data for {env_name}')
        self.env = gym.make(env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim)
        self.replay_buffer.convert_D4RL(d4rl.qlearning_dataset(self.env))

        # Initilaize saving settings
        self.save_dir = f"{model_dir}/{env_name}/s{seed}"
        # self.save_dir = f"{model_dir}/{env_name}"
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

    def load_new(self, filename):
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

    def train_new(self):
        (inputs, targets, holdout_inputs, holdout_targets, self.obs_mean, self.obs_std) = get_training_data(self.replay_buffer, self.ensemble_num, self.holdout_ratio)
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
                train_loss = jnp.mean(mse_loss * inv_var + log_var, axis=-1).sum() 
                delta_log_var = self.model.apply({"params": params}, method=self.model.delta_log_var)
                train_loss += 0.01 * delta_log_var
                return train_loss, {"mse_loss": mse_loss.mean(),
                                    "var_loss": log_var.mean(),
                                    "train_loss": train_loss,
                                    "delta_log_var": delta_log_var}
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
                mse_loss = jnp.square(mu - y)  # (7, 12) ==> (7,)
                # mse_loss = jnp.mean(jnp.square(mu - y), axis=-1)  # (7, 12) ==> (7,)
                # return mse_loss, {"reward_loss": reward_loss, "state_loss": state_loss, "mse_loss": mse_loss}
                eval_loss = jnp.mean(mse_loss[:, :-1], axis=-1) + mse_loss[:, -1]
                return eval_loss, {"reward_loss": reward_loss, "state_loss": state_loss, "mse_loss": mse_loss}
            loss_fn = jax.vmap(loss_fn, in_axes=(None, 1, 1))
            loss, log_info = loss_fn(model_state.params, batch_inputs, batch_targets)
            log_info = jax.tree_map(functools.partial(jnp.mean, axis=0), log_info)
            return loss, log_info

        for epoch in trange(self.epochs):
            shuffled_idxs = np.concatenate([np.random.permutation(np.arange(
                inputs.shape[0])).reshape(1, -1) for _ in range(self.ensemble_num)], axis=0)  # (7, 1000000)
            train_loss, mse_loss, var_loss, delta_log_var = [], [], [], []
            for i in range(batch_num):
                batch_idxs = shuffled_idxs[:, i*self.batch_size:(i+1)*self.batch_size]
                batch_inputs = inputs[batch_idxs]    # (7, 256, 14)
                batch_targets = targets[batch_idxs]  # (7, 256, 12)
                self.model_state, log_info = train_step(self.model_state, batch_inputs, batch_targets)
                train_loss.append(log_info["train_loss"].item())
                mse_loss.append(log_info["mse_loss"].item())
                var_loss.append(log_info["var_loss"].item())
                delta_log_var.append(log_info["delta_log_var"].item())

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
            if epoch > 20 and patience > self.max_patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

            res.append((epoch, sum(train_loss)/batch_num, sum(mse_loss)/batch_num, sum(var_loss)/batch_num, mean_val_loss))
            print(f"Epoch #{epoch+1}: "
                  f"train_loss={sum(train_loss)/batch_num:.3f}\t"
                  f"mse_loss={sum(mse_loss)/batch_num:.3f}\t"
                  f"var_loss={sum(var_loss)/batch_num:.3f}\t"
                  f"delta_log_var={sum(delta_log_var)/batch_num:.3f}\t"
                  f"val_loss={mean_val_loss:.3f}\t"
                  f"val_rew_loss={val_info['reward_loss']:.3f}\t"
                  f"val_state_loss={val_info['state_loss']:.3f}")

            if (epoch+1) in [10, 50, 100, 150]:
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

    def load(self, filename):
        with open(f"{filename}/dynamics_model.ckpt", "rb") as f:
            model_params = serialization.from_bytes(
                self.model_state.params, f.read())
        self.model_state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=model_params,
            tx=optax.adamw(learning_rate=self.lr,
            weight_decay=self.weight_decay))
        elite_idx = np.loadtxt(f'{filename}/elite_models.txt', dtype=np.int32)[:self.elite_num]
        self.elite_mask = np.eye(self.ensemble_num)[elite_idx, :]
        normalize_stat = np.load(f'{filename}/normalize_stat.npz')
        self.obs_mean = normalize_stat['obs_mean'].squeeze()
        self.obs_std = normalize_stat['obs_std'].squeeze()

    def train(self):
        (inputs, targets, holdout_inputs, holdout_targets, self.obs_mean, self.obs_std) = get_training_data(self.replay_buffer, self.ensemble_num, self.holdout_ratio)
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
                train_loss = jnp.mean(mse_loss * inv_var + log_var, axis=-1).sum() 
                return train_loss, {"mse_loss": mse_loss.mean(),
                                    "var_loss": log_var.mean(),
                                    "train_loss": train_loss}
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

        for epoch in trange(self.epochs):
            shuffled_idxs = np.concatenate([np.random.permutation(np.arange(
                inputs.shape[0])).reshape(1, -1) for _ in range(self.ensemble_num)], axis=0)  # (7, 1000000)
            train_loss, mse_loss, var_loss, delta_log_var = [], [], [], []
            for i in range(batch_num):
                batch_idxs = shuffled_idxs[:, i*self.batch_size:(i+1)*self.batch_size]
                batch_inputs = inputs[batch_idxs]    # (7, 256, 14)
                batch_targets = targets[batch_idxs]  # (7, 256, 12)
                self.model_state, log_info = train_step(self.model_state, batch_inputs, batch_targets)
                train_loss.append(log_info["train_loss"].item())
                mse_loss.append(log_info["mse_loss"].item())
                var_loss.append(log_info["var_loss"].item())
                delta_log_var.append(log_info["delta_log_var"].item())

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
            if epoch > 20 and patience > self.max_patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

            res.append((epoch, sum(train_loss)/batch_num, sum(mse_loss)/batch_num, sum(var_loss)/batch_num, mean_val_loss))
            print(f"Epoch #{epoch+1}: "
                  f"train_loss={sum(train_loss)/batch_num:.3f}\t"
                  f"mse_loss={sum(mse_loss)/batch_num:.3f}\t"
                  f"var_loss={sum(var_loss)/batch_num:.3f}\t"
                  f"delta_log_var={sum(delta_log_var)/batch_num:.3f}\t"
                  f"val_loss={mean_val_loss:.3f}\t"
                  f"val_rew_loss={val_info['reward_loss']:.3f}\t"
                  f"val_state_loss={val_info['state_loss']:.3f}")

            if (epoch+1) in [10, 50, 100, 150]:
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

    def rollout_noise(self, rollout_key, params, observations, actions, model_masks):
        @jax.jit
        def rollout_fn(key, observation, action, model_mask):
            x = jnp.concatenate([observation, action], axis=-1).reshape(1, -1)
            model_mu, model_log_var = self.model.apply({"params": params}, x)
            model_std = jnp.exp(0.5*model_log_var)
            model_noise = jax.random.normal(key, shape=model_mu.shape) * model_std * self.noise_scale
            observation_mu, reward_mu = jnp.split(model_mu, [self.obs_dim], axis=-1)
            observation_noise, reward_noise = jnp.split(model_noise, [self.obs_dim], axis=-1)
            model_next_observation = observation + jnp.sum(model_mask * (observation_mu + observation_noise), axis=0)
            model_reward = jnp.sum(model_mask * (reward_mu + reward_noise), axis=0)
            return model_next_observation, model_reward
        keys = jnp.stack(jax.random.split(rollout_key, num=actions.shape[0]))
        next_observations, rewards = jax.vmap(rollout_fn, in_axes=(0, 0, 0, 0))(keys, observations, actions, model_masks)
        next_observations = self.denormalize(next_observations)
        return next_observations, rewards

    def rollout(self, params, observations, actions, model_masks):
        @jax.jit
        def rollout_fn(observation, action, model_mask):
            x = jnp.concatenate([observation, action], axis=-1).reshape(1, -1)
            model_mu, _ = self.model.apply({"params": params}, x)
            observation_mu, reward_mu = jnp.split(model_mu, [self.obs_dim], axis=-1)
            model_next_observation = observation + jnp.sum(model_mask * observation_mu, axis=0)
            model_reward = jnp.sum(model_mask * reward_mu, axis=0)
            return model_next_observation, model_reward
        next_observations, rewards = jax.vmap(rollout_fn, in_axes=(0, 0, 0))(observations, actions, model_masks)
        next_observations = self.denormalize(next_observations)
        return next_observations, rewards

    def step_noise(self, key, observations, actions):
        model_idx = jax.random.randint(key, shape=(actions.shape[0],), minval=0, maxval=self.elite_num)
        model_masks = self.elite_mask[model_idx].reshape(-1, self.ensemble_num, 1)
        next_observations, rewards = self.rollout_noise(key, self.model_state.params, observations, actions, model_masks)
        terminals = self.static_fn.termination_fn(observations, actions, next_observations)
        return next_observations, rewards.squeeze(), terminals.squeeze()

    def step(self, key, observations, actions):
        model_idx = jax.random.randint(key, shape=(actions.shape[0],), minval=0, maxval=self.elite_num)
        model_masks = self.elite_mask[model_idx].reshape(-1, self.ensemble_num, 1)
        next_observations, rewards = self.rollout(self.model_state.params, observations, actions, model_masks)
        terminals = self.static_fn.termination_fn(observations, actions, next_observations)
        return next_observations, rewards.squeeze(), terminals.squeeze()

    def normalize(self, observations):
        new_observations = (observations - self.obs_mean) / self.obs_std
        return new_observations

    def denormalize(self, observations):
        new_observations = observations * self.obs_std + self.obs_mean
        return new_observations


class COMBOAgent:
    def __init__(self,
                 env_name: str = "hopper-medium-v2",
                 obs_dim: int = 11,
                 act_dim: int = 3,
                 max_action: float = 1.0,
                 seed: int = 42,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 lr_critic: float = 3e-4,
                 lr_actor: float = 1e-4,
                 target_entropy: float = None,
                 backup_entropy: bool = False,
                 num_random: int = 10,
                 min_q_weight: float = 3.0,

                 # COMBO
                 horizon: int = 5,
                 noise_scale: float = 0.0,
                 real_ratio: float = 0.5,
                 ensemble_num: int = 7,
                 elite_num: int = 5,
                 max_patience: int = 5,
                 batch_size: int = 256,
                 rollout_batch_size: int = 10000,
                 holdout_ratio: float = 0.1,
                 rollout_random: bool = False,
                 hidden_dims: Sequence[int] = (256, 256, 256),
                 initializer: str = 'orthogonal'):
                 

        self.update_step = 0
        self.max_action = max_action
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.backup_entropy = backup_entropy
        if target_entropy is None:
            self.target_entropy = -act_dim
        else:
            self.target_entropy = target_entropy

        # Initialize random keys
        self.rng = jax.random.PRNGKey(seed)
        self.rng, self.rollout_rng, actor_key, critic_key = jax.random.split(self.rng, 4)

        # Dummy inputs
        dummy_obs = jnp.ones([1, obs_dim], dtype=jnp.float32)
        dummy_act = jnp.ones([1, act_dim], dtype=jnp.float32)

        # Initialize the Actor
        self.actor = Actor(act_dim=act_dim, max_action=max_action,
                           hidden_dims=hidden_dims, initializer=initializer)
        actor_params = self.actor.init(actor_key, actor_key, dummy_obs)["params"]
        self.actor_state = train_state.TrainState.create(
            apply_fn=Actor.apply,
            params=actor_params,
            tx=optax.chain(optax.clip(1.0), optax.adam(learning_rate=lr_actor)))

        # Initialize the Critic
        self.critic = DoubleCritic(hidden_dims=hidden_dims, initializer=initializer)
        critic_params = self.critic.init(critic_key, dummy_obs, dummy_act)["params"]
        self.critic_target_params = critic_params
        self.critic_state = train_state.TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=optax.chain(optax.clip(1.0), optax.adam(learning_rate=lr_critic)))

        # Entropy tuning
        self.rng, alpha_key = jax.random.split(self.rng, 2)
        self.log_alpha = Scalar(0.0)
        self.alpha_state = train_state.TrainState.create(
            apply_fn=None,
            params=self.log_alpha.init(alpha_key)["params"],
            tx=optax.chain(optax.clip(1.0), optax.adam(lr_actor))
        )

        # Initialize the Dynamics Model
        self.model = DynamicsModel(env_name=env_name,
                                   seed=seed,
                                   ensemble_num=ensemble_num,   
                                   elite_num=elite_num,
                                   noise_scale=noise_scale)

        # COMBO parameters
        self.horizon = horizon
        self.num_random = num_random
        self.min_q_weight = min_q_weight
        self.rollout_random = rollout_random
        self.batch_size = batch_size
        self.rollout_batch_size = rollout_batch_size

        self.max_patience = max_patience
        self.holdout_ratio = holdout_ratio
        self.ensemble_num = ensemble_num
        self.real_ratio = real_ratio 
        self.real_batch_size = int(real_ratio * batch_size)
        self.model_batch_size = batch_size - self.real_batch_size
        self.real_batch_ratio = self.batch_size / self.real_batch_size
        self.model_batch_ratio = self.batch_size / self.model_batch_size
        self.masks_real = self.real_batch_ratio * np.concatenate([np.ones(self.real_batch_size),
                                                                  np.zeros(self.model_batch_size)])
        if env_name in ['hopper-medium-v2', 'hopper-medium-replay-v2']:
            self.masks_model = self.model_batch_ratio * np.concatenate([np.zeros(self.real_batch_size),
                                                                        np.ones(self.model_batch_size)])
        else:
            self.masks_model = np.ones(self.batch_size)

    @functools.partial(jax.jit, static_argnames=("self"))
    def select_action(self, params: FrozenDict, rng: Any, observation: np.ndarray, eval_mode: bool = False) -> jnp.ndarray:
        rng, sample_rng = jax.random.split(rng)
        mean_action, sampled_action, _ = self.actor.apply({"params": params}, sample_rng, observation)
        return rng, jnp.where(eval_mode, mean_action, sampled_action)

    @functools.partial(jax.jit, static_argnames=("self"), device=jax.devices("cpu")[0])
    def eval_select_action(self, params: FrozenDict, rng: Any, observation: np.ndarray) -> jnp.ndarray:
        mean_action, _, _ = self.actor.apply({"params": params}, rng, observation)
        return rng, mean_action

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self,
                   batch: Batch,
                   key: jnp.ndarray,
                   alpha_state: train_state.TrainState,
                   actor_state: train_state.TrainState,
                   critic_state: train_state.TrainState,
                   critic_target_params: FrozenDict):

        # For use in loss_fn without apply gradients
        frozen_actor_params = actor_state.params
        frozen_critic_params = critic_state.params

        def loss_fn(alpha_params: FrozenDict,
                    actor_params: FrozenDict,
                    critic_params: FrozenDict, 
                    rng: jnp.ndarray,
                    observation: jnp.ndarray,
                    action: jnp.ndarray,
                    reward: jnp.ndarray,
                    next_observation: jnp.ndarray,
                    discount: jnp.ndarray,
                    mask_real: jnp.ndarray,
                    mask_model: jnp.ndarray):
            """compute loss for a single transition"""
            rng, rng1, rng2 = jax.random.split(rng, 3)

            # Sample actions with Actor
            _, sampled_action, logp = self.actor.apply({"params": actor_params}, rng1, observation)

            # Alpha loss: stop gradient to avoid affect Actor parameters
            log_alpha = self.log_alpha.apply({"params": alpha_params})
            alpha_loss = -log_alpha * jax.lax.stop_gradient(logp + self.target_entropy)
            alpha = jnp.exp(log_alpha)
            alpha = jax.lax.stop_gradient(alpha)  # stop gradient to avoid affect Alpha parameters

            # We use frozen_params so that gradients can flow back to the actor without being used to update the critic.
            sampled_q1, sampled_q2 = self.critic.apply({"params": frozen_critic_params}, observation, sampled_action)
            sampled_q = jnp.minimum(sampled_q1, sampled_q2)

            # Actor loss
            actor_loss = alpha * logp - sampled_q

            # Critic loss
            q1, q2 = self.critic.apply({"params": critic_params}, observation, action)

            # Use frozen_actor_params to avoid affect Actor parameters
            _, next_action, logp_next_action = self.actor.apply({"params": frozen_actor_params}, rng2, next_observation)
            next_q1, next_q2 = self.critic.apply({"params": critic_target_params}, next_observation, next_action)
            next_q = jnp.minimum(next_q1, next_q2)
            if self.backup_entropy:
                next_q -= alpha * logp_next_action
            target_q = reward + self.gamma * discount * next_q
            critic_loss1 = 0.5 * (q1 - target_q)**2
            critic_loss2 = 0.5 * (q2 - target_q)**2
            critic_loss = critic_loss1 + critic_loss2

            # COMBO CQL loss
            rng3, rng4 = jax.random.split(rng, 2)
            cql_random_actions = jax.random.uniform(
                rng3, shape=(self.num_random, self.act_dim), minval=-self.max_action, maxval=self.max_action)

            # repeat next observations
            repeat_observations = jnp.repeat(jnp.expand_dims(observation, axis=0),
                                             repeats=self.num_random, axis=0)          # (10, 11)

            # sample actions with actor
            _, cql_sampled_actions, cql_logp = self.actor.apply(
                {"params": frozen_actor_params}, rng4, repeat_observations)            # (10, 3),  (10,)

            # random q values
            cql_random_q1, cql_random_q2 = self.critic.apply({"params": critic_params},
                                                             repeat_observations,
                                                             cql_random_actions)       # (10, 1), (10, 1)

            # cql q values
            cql_q1, cql_q2 = self.critic.apply({"params": critic_params}, repeat_observations, cql_sampled_actions)

            random_density = np.log(0.5 ** self.act_dim)
            cql_concat_q1 = jnp.concatenate([cql_random_q1 - random_density, cql_q1 - cql_logp])
            cql_concat_q2 = jnp.concatenate([cql_random_q2 - random_density, cql_q2 - cql_logp])

            ood_q1 = jax.scipy.special.logsumexp(cql_concat_q1)
            ood_q2 = jax.scipy.special.logsumexp(cql_concat_q2)

            # compute logsumexp loss w.r.t model_states 
            cql1_loss = (ood_q1*mask_model - q1*mask_real) * self.min_q_weight
            cql2_loss = (ood_q2*mask_model - q2*mask_real) * self.min_q_weight

            total_loss = alpha_loss + actor_loss + critic_loss + cql1_loss + cql2_loss
            log_info = {
                "critic_loss1": critic_loss1,
                "critic_loss2": critic_loss2,
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "alpha_loss": alpha_loss,
                "cql1_loss": cql1_loss,
                "cql2_loss": cql2_loss, 
                "q1": q1,
                "q2": q2,
                "target_q": target_q,
                "sampled_q": sampled_q,
                "ood_q1": ood_q1,
                "ood_q2": ood_q2,
                "cql_q1": cql_q1.mean(),
                "cql_q2": cql_q2.mean(),
                "random_q1": cql_random_q1.mean(),
                "random_q2": cql_random_q2.mean(),
                "alpha": alpha,
                "logp": logp,
                "min_q_weight": self.min_q_weight,
                "logp_next_action": logp_next_action
            }

            return total_loss, log_info

        grad_fn = jax.vmap(
            jax.value_and_grad(loss_fn, argnums=(0, 1, 2), has_aux=True),
            in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, 0, 0))
        keys = jnp.stack(jax.random.split(key, num=batch.actions.shape[0]))
        (_, log_info), gradients = grad_fn(alpha_state.params,
                                           actor_state.params,
                                           critic_state.params,
                                           keys,
                                           batch.observations,
                                           batch.actions,
                                           batch.rewards,
                                           batch.next_observations,
                                           batch.discounts,
                                           self.masks_real,
                                           self.masks_model)
        gradients = jax.tree_map(functools.partial(jnp.mean, axis=0), gradients)
        extra_log_info = {
            'q1_min': log_info['q1'].min(),
            'q1_max': log_info['q1'].max(),
            'q1_std': log_info['q1'].std(),
            'q2_min': log_info['q2'].min(),
            'q2_max': log_info['q2'].max(),
            'q2_std': log_info['q2'].std(),
            'target_q_min': log_info['target_q'].min(),
            'target_q_max': log_info['target_q'].max(),
            'target_q_std': log_info['target_q'].std(),
            'ood_q1_min': log_info['ood_q1'].min(),
            'ood_q1_max': log_info['ood_q1'].max(),
            'ood_q1_std': log_info['ood_q1'].std(),
            'ood_q2_min': log_info['ood_q2'].min(),
            'ood_q2_max': log_info['ood_q2'].max(),
            'ood_q2_std': log_info['ood_q2'].std(),
            'critic_loss_min': log_info['critic_loss'].min(),
            'critic_loss_max': log_info['critic_loss'].max(),
            'critic_loss_std': log_info['critic_loss'].std(),
            'critic_loss1_min': log_info['critic_loss1'].min(),
            'critic_loss1_max': log_info['critic_loss1'].max(),
            'critic_loss1_std': log_info['critic_loss1'].std(),
            'critic_loss2_min': log_info['critic_loss2'].min(),
            'critic_loss2_max': log_info['critic_loss2'].max(),
            'critic_loss2_std': log_info['critic_loss2'].std(),
            'real_critic_loss': log_info['critic_loss'][:self.real_batch_size].mean(),
            'fake_critic_loss': log_info['critic_loss'][self.real_batch_size:].mean(),
            'real_critic_loss_max': log_info['critic_loss'][:self.real_batch_size].max(),
            'fake_critic_loss_min': log_info['critic_loss'][self.real_batch_size:].min(),
            'cql1_loss_min': log_info['cql1_loss'].min(),
            'cql1_loss_max': log_info['cql1_loss'].max(),
            'cql1_loss_std': log_info['cql1_loss'].std(),
            'cql2_loss_min': log_info['cql2_loss'].min(),
            'cql2_loss_max': log_info['cql2_loss'].max(),
            'cql2_loss_std': log_info['cql2_loss'].std(),
        }
        log_info = jax.tree_map(functools.partial(jnp.mean, axis=0), log_info)
        log_info.update(extra_log_info)
        alpha_grads, actor_grads, critic_grads = gradients

        # Update TrainState
        new_alpha_state = alpha_state.apply_gradients(grads=alpha_grads)
        new_actor_state = actor_state.apply_gradients(grads=actor_grads)
        new_critic_state = critic_state.apply_gradients(grads=critic_grads)
        new_critic_target_params = target_update(new_critic_state.params, critic_target_params, self.tau)
        return new_alpha_state, new_actor_state, new_critic_state, new_critic_target_params, log_info

    def update(self, replay_buffer, model_buffer):
        select_action = jax.vmap(self.select_action, in_axes=(None, 0, 0, None))
        if self.update_step % 1000 == 0:
            observations = replay_buffer.sample(self.rollout_batch_size).observations
            sample_rng = jnp.stack(jax.random.split(self.rollout_rng, num=self.rollout_batch_size))
            for t in range(self.horizon):
                self.rollout_rng, rollout_key = jax.random.split(self.rollout_rng, 2)
                if self.rollout_random:
                    actions = np.random.uniform(low=-self.max_action, high=self.max_action,
                                                size=(len(observations), self.act_dim))
                else:
                    sample_rng, actions = select_action(self.actor_state.params, sample_rng, observations, False)

                # normalize states and actions
                normalized_observations = self.model.normalize(observations)
                next_observations, rewards, dones = self.model.step(rollout_key, normalized_observations, actions)
                nonterminal_mask = ~dones
                model_buffer.add_batch(observations,
                                       actions,
                                       next_observations,
                                       rewards,
                                       dones)
                if nonterminal_mask.sum() == 0:
                    print(f'[ Model Rollout ] Breaking early {nonterminal_mask.shape}')
                    break
                observations = next_observations[nonterminal_mask]
                sample_rng = sample_rng[nonterminal_mask]

        # sample from real & model buffer
        real_batch = replay_buffer.sample(self.real_batch_size)
        model_batch = model_buffer.sample(self.model_batch_size)
        concat_batch = Batch(
            observations=np.concatenate([real_batch.observations, model_batch.observations], axis=0),
            actions = np.concatenate([real_batch.actions, model_batch.actions], axis=0),
            rewards = np.concatenate([real_batch.rewards, model_batch.rewards], axis=0),
            discounts = np.concatenate([real_batch.discounts, model_batch.discounts], axis=0),
            next_observations = np.concatenate([real_batch.next_observations, model_batch.next_observations], axis=0)
        )

        # CQL training with COMBO
        self.rng, key = jax.random.split(self.rng)
        (self.alpha_state, self.actor_state, self.critic_state, self.critic_target_params, log_info) = self.train_step(
            concat_batch, key, self.alpha_state, self.actor_state, self.critic_state, self.critic_target_params)

        log_info['real_batch_rewards'] = real_batch.rewards.sum()
        log_info['real_batch_rewards_min'] = real_batch.rewards.min()
        log_info['real_batch_rewards_max'] = real_batch.rewards.max()
        log_info['real_batch_actions'] = abs(real_batch.actions).reshape(-1).sum()
        log_info['real_batch_observations'] = abs(real_batch.observations).mean(0).sum()
        log_info['real_batch_discounts'] = real_batch.discounts.sum()
        log_info['model_batch_rewards'] = model_batch.rewards.sum()
        log_info['model_batch_rewards_min'] = model_batch.rewards.min()
        log_info['model_batch_rewards_max'] = model_batch.rewards.max()
        log_info['model_batch_actions'] = abs(model_batch.actions).reshape(-1).sum()
        log_info['model_batch_observations'] = abs(model_batch.observations).mean(0).sum()
        log_info['model_batch_discounts'] = model_batch.discounts.sum()
        log_info['model_buffer_size'] = model_buffer.size
        log_info['model_buffer_ptr'] = model_buffer.ptr
        self.update_step += 1
        return log_info

    def save(self, fname: str, cnt: int):
        checkpoints.save_checkpoint(fname, self.actor_state, cnt, prefix="actor_", keep=20, overwrite=True)
        checkpoints.save_checkpoint(fname, self.critic_state, cnt, prefix="critic_", keep=20, overwrite=True)

    def load(self, ckpt_dir, step):
        self.actor_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir,
                                                          target=self.actor_state,
                                                          step=step,
                                                          prefix="actor_")
        self.critic_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir,
                                                           target=self.critic_state,
                                                           step=step,
                                                           prefix="critic_")



# only use model data to learn the critic
class COMBOAgent_nocql:
    def __init__(self,
                 env_name: str = "hopper-medium-v2",
                 obs_dim: int = 11,
                 act_dim: int = 3,
                 max_action: float = 1.0,
                 seed: int = 42,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 lr_critic: float = 3e-4,
                 lr_actor: float = 1e-4,
                 target_entropy: float = None,
                 backup_entropy: bool = False,
                 num_random: int = 10,
                 min_q_weight: float = 3.0,

                 # COMBO
                 horizon: int = 5,
                 noise_scale: float = 0.0,
                 real_ratio: float = 0.5,
                 ensemble_num: int = 7,
                 elite_num: int = 5,
                 max_patience: int = 5,
                 batch_size: int = 256,
                 rollout_batch_size: int = 10000,
                 holdout_ratio: float = 0.1,
                 rollout_random: bool = False,
                 hidden_dims: Sequence[int] = (256, 256, 256),
                 initializer: str = 'orthogonal'):
                 

        self.update_step = 0
        self.max_action = max_action
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.backup_entropy = backup_entropy
        if target_entropy is None:
            self.target_entropy = -act_dim
        else:
            self.target_entropy = target_entropy

        # Initialize random keys
        self.rng = jax.random.PRNGKey(seed)
        self.rng, self.rollout_rng, actor_key, critic_key = jax.random.split(self.rng, 4)

        # Dummy inputs
        dummy_obs = jnp.ones([1, obs_dim], dtype=jnp.float32)
        dummy_act = jnp.ones([1, act_dim], dtype=jnp.float32)

        # Initialize the Actor
        self.actor = Actor(act_dim=act_dim, max_action=max_action,
                           hidden_dims=hidden_dims, initializer=initializer)
        actor_params = self.actor.init(actor_key, actor_key, dummy_obs)["params"]
        self.actor_state = train_state.TrainState.create(
            apply_fn=Actor.apply,
            params=actor_params,
            tx=optax.chain(optax.clip(1.0), optax.adam(learning_rate=lr_actor)))

        # Initialize the Critic
        self.critic = DoubleCritic(hidden_dims=hidden_dims, initializer=initializer)
        critic_params = self.critic.init(critic_key, dummy_obs, dummy_act)["params"]
        self.critic_target_params = critic_params
        self.critic_state = train_state.TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=optax.chain(optax.clip(1.0), optax.adam(learning_rate=lr_critic)))

        # Entropy tuning
        self.rng, alpha_key = jax.random.split(self.rng, 2)
        self.log_alpha = Scalar(0.0)
        self.alpha_state = train_state.TrainState.create(
            apply_fn=None,
            params=self.log_alpha.init(alpha_key)["params"],
            tx=optax.chain(optax.clip(1.0), optax.adam(lr_actor))
        )

        # Initialize the Dynamics Model
        self.model = DynamicsModel(env_name=env_name,
                                   seed=seed,
                                   ensemble_num=ensemble_num,   
                                   elite_num=elite_num,
                                   noise_scale=noise_scale)

        # COMBO parameters
        self.horizon = horizon
        self.num_random = num_random
        self.min_q_weight = min_q_weight
        self.rollout_random = rollout_random
        self.batch_size = batch_size
        self.rollout_batch_size = rollout_batch_size

        self.max_patience = max_patience
        self.holdout_ratio = holdout_ratio
        self.ensemble_num = ensemble_num
        self.real_ratio = real_ratio 
        self.real_batch_size = int(real_ratio * batch_size)
        self.model_batch_size = batch_size - self.real_batch_size
        self.real_batch_ratio = self.batch_size / self.real_batch_size
        self.model_batch_ratio = self.batch_size / self.model_batch_size
        self.masks_real = self.real_batch_ratio * np.concatenate([np.ones(self.real_batch_size),
                                                                  np.zeros(self.model_batch_size)])
        # if env_name in ['hopper-medium-v2', 'hopper-medium-replay-v2', 'halfcheetah-medium-v2', "halfcheetah-medium-replay-v2"]:
        if env_name in ['hopper-medium-v2', 'hopper-medium-replay-v2']:
            self.masks_model = self.model_batch_ratio * np.concatenate([np.zeros(self.real_batch_size), np.ones(self.model_batch_size)])
        else:
            self.masks_model = np.ones(self.batch_size)

    @functools.partial(jax.jit, static_argnames=("self"))
    def select_action(self, params: FrozenDict, rng: Any, observation: np.ndarray, eval_mode: bool = False) -> jnp.ndarray:
        rng, sample_rng = jax.random.split(rng)
        mean_action, sampled_action, _ = self.actor.apply({"params": params}, sample_rng, observation)
        return rng, jnp.where(eval_mode, mean_action, sampled_action)

    @functools.partial(jax.jit, static_argnames=("self"), device=jax.devices("cpu")[0])
    def eval_select_action(self, params: FrozenDict, rng: Any, observation: np.ndarray) -> jnp.ndarray:
        mean_action, _, _ = self.actor.apply({"params": params}, rng, observation)
        return rng, mean_action

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self,
                   batch: Batch,
                   key: jnp.ndarray,
                   alpha_state: train_state.TrainState,
                   actor_state: train_state.TrainState,
                   critic_state: train_state.TrainState,
                   critic_target_params: FrozenDict):

        # For use in loss_fn without apply gradients
        frozen_actor_params = actor_state.params
        frozen_critic_params = critic_state.params

        def loss_fn(alpha_params: FrozenDict,
                    actor_params: FrozenDict,
                    critic_params: FrozenDict, 
                    rng: jnp.ndarray,
                    observation: jnp.ndarray,
                    action: jnp.ndarray,
                    reward: jnp.ndarray,
                    next_observation: jnp.ndarray,
                    discount: jnp.ndarray,
                    mask_real: jnp.ndarray,
                    mask_model: jnp.ndarray):
            """compute loss for a single transition"""
            rng, rng1, rng2 = jax.random.split(rng, 3)

            # Sample actions with Actor
            _, sampled_action, logp = self.actor.apply({"params": actor_params}, rng1, observation)

            # Alpha loss: stop gradient to avoid affect Actor parameters
            log_alpha = self.log_alpha.apply({"params": alpha_params})
            alpha_loss = -log_alpha * jax.lax.stop_gradient(logp + self.target_entropy)
            alpha = jnp.exp(log_alpha)
            alpha = jax.lax.stop_gradient(alpha)  # stop gradient to avoid affect Alpha parameters

            # We use frozen_params so that gradients can flow back to the actor without being used to update the critic.
            sampled_q1, sampled_q2 = self.critic.apply({"params": frozen_critic_params}, observation, sampled_action)
            sampled_q = jnp.minimum(sampled_q1, sampled_q2)

            # Actor loss
            actor_loss = alpha * logp - sampled_q

            # Critic loss
            q1, q2 = self.critic.apply({"params": critic_params}, observation, action)

            # Use frozen_actor_params to avoid affect Actor parameters
            _, next_action, logp_next_action = self.actor.apply({"params": frozen_actor_params}, rng2, next_observation)
            next_q1, next_q2 = self.critic.apply({"params": critic_target_params}, next_observation, next_action)
            next_q = jnp.minimum(next_q1, next_q2)
            if self.backup_entropy:
                next_q -= alpha * logp_next_action
            target_q = reward + self.gamma * discount * next_q
            critic_loss1 = 0.5 * (q1 - target_q)**2
            critic_loss2 = 0.5 * (q2 - target_q)**2
            critic_loss = critic_loss1 + critic_loss2

            total_loss = alpha_loss + actor_loss + critic_loss
            log_info = {
                "critic_loss1": critic_loss1,
                "critic_loss2": critic_loss2,
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "alpha_loss": alpha_loss,
                "cql1_loss": 0,
                "cql2_loss": 0, 
                "q1": q1,
                "q2": q2,
                "target_q": target_q,
                "sampled_q": sampled_q,
                "ood_q1": 0,
                "ood_q2": 0,
                "cql_q1": 0,
                "cql_q2": 0,
                "random_q1": 0,
                "random_q2": 0,
                "alpha": alpha,
                "logp": logp,
                "min_q_weight": self.min_q_weight,
                "logp_next_action": logp_next_action
            }

            return total_loss, log_info

        grad_fn = jax.vmap(
            jax.value_and_grad(loss_fn, argnums=(0, 1, 2), has_aux=True),
            in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, 0, 0))
        keys = jnp.stack(jax.random.split(key, num=batch.actions.shape[0]))
        (_, log_info), gradients = grad_fn(alpha_state.params,
                                           actor_state.params,
                                           critic_state.params,
                                           keys,
                                           batch.observations,
                                           batch.actions,
                                           batch.rewards,
                                           batch.next_observations,
                                           batch.discounts,
                                           self.masks_real,
                                           self.masks_model)
        gradients = jax.tree_map(functools.partial(jnp.mean, axis=0), gradients)
        extra_log_info = {
            'q1_min': log_info['q1'].min(),
            'q1_max': log_info['q1'].max(),
            'q1_std': log_info['q1'].std(),
            'q2_min': log_info['q2'].min(),
            'q2_max': log_info['q2'].max(),
            'q2_std': log_info['q2'].std(),
            'target_q_min': log_info['target_q'].min(),
            'target_q_max': log_info['target_q'].max(),
            'target_q_std': log_info['target_q'].std(),
            'ood_q1_min': log_info['ood_q1'].min(),
            'ood_q1_max': log_info['ood_q1'].max(),
            'ood_q1_std': log_info['ood_q1'].std(),
            'ood_q2_min': log_info['ood_q2'].min(),
            'ood_q2_max': log_info['ood_q2'].max(),
            'ood_q2_std': log_info['ood_q2'].std(),
            'critic_loss_min': log_info['critic_loss'].min(),
            'critic_loss_max': log_info['critic_loss'].max(),
            'critic_loss_std': log_info['critic_loss'].std(),
            'critic_loss1_min': log_info['critic_loss1'].min(),
            'critic_loss1_max': log_info['critic_loss1'].max(),
            'critic_loss1_std': log_info['critic_loss1'].std(),
            'critic_loss2_min': log_info['critic_loss2'].min(),
            'critic_loss2_max': log_info['critic_loss2'].max(),
            'critic_loss2_std': log_info['critic_loss2'].std(),
            'real_critic_loss': log_info['critic_loss'][:self.real_batch_size].mean(),
            'fake_critic_loss': log_info['critic_loss'][self.real_batch_size:].mean(),
            'real_critic_loss_max': log_info['critic_loss'][:self.real_batch_size].max(),
            'fake_critic_loss_min': log_info['critic_loss'][self.real_batch_size:].min(),
            'cql1_loss_min': log_info['cql1_loss'].min(),
            'cql1_loss_max': log_info['cql1_loss'].max(),
            'cql1_loss_std': log_info['cql1_loss'].std(),
            'cql2_loss_min': log_info['cql2_loss'].min(),
            'cql2_loss_max': log_info['cql2_loss'].max(),
            'cql2_loss_std': log_info['cql2_loss'].std(),
        }
        log_info = jax.tree_map(functools.partial(jnp.mean, axis=0), log_info)
        log_info.update(extra_log_info)
        alpha_grads, actor_grads, critic_grads = gradients

        # Update TrainState
        new_alpha_state = alpha_state.apply_gradients(grads=alpha_grads)
        new_actor_state = actor_state.apply_gradients(grads=actor_grads)
        new_critic_state = critic_state.apply_gradients(grads=critic_grads)
        new_critic_target_params = target_update(new_critic_state.params, critic_target_params, self.tau)
        return new_alpha_state, new_actor_state, new_critic_state, new_critic_target_params, log_info

    def update(self, replay_buffer, model_buffer):
        select_action = jax.vmap(self.select_action, in_axes=(None, 0, 0, None))
        if self.update_step % 1000 == 0:
            observations = replay_buffer.sample(self.rollout_batch_size).observations
            sample_rng = jnp.stack(jax.random.split(self.rollout_rng, num=self.rollout_batch_size))
            for t in range(self.horizon):
                self.rollout_rng, rollout_key = jax.random.split(self.rollout_rng, 2)
                if self.rollout_random:
                    actions = np.random.uniform(low=-self.max_action, high=self.max_action,
                                                size=(len(observations), self.act_dim))
                else:
                    sample_rng, actions = select_action(self.actor_state.params, sample_rng, observations, False)

                # normalize states and actions
                normalized_observations = self.model.normalize(observations)
                next_observations, rewards, dones = self.model.step(rollout_key, normalized_observations, actions)
                nonterminal_mask = ~dones
                model_buffer.add_batch(observations, actions, next_observations, rewards, dones)
                if nonterminal_mask.sum() == 0: break
                observations = next_observations[nonterminal_mask]
                sample_rng = sample_rng[nonterminal_mask]

        # sample from real & model buffer
        real_batch = replay_buffer.sample(self.real_batch_size)
        model_batch = model_buffer.sample(self.model_batch_size)
        concat_batch = Batch(
            observations=np.concatenate([real_batch.observations, model_batch.observations], axis=0),
            actions = np.concatenate([real_batch.actions, model_batch.actions], axis=0),
            rewards = np.concatenate([real_batch.rewards, model_batch.rewards], axis=0),
            discounts = np.concatenate([real_batch.discounts, model_batch.discounts], axis=0),
            next_observations = np.concatenate([real_batch.next_observations, model_batch.next_observations], axis=0)
        )

        # CQL training with COMBO
        self.rng, key = jax.random.split(self.rng)
        (self.alpha_state, self.actor_state, self.critic_state, self.critic_target_params, log_info) = self.train_step(
            concat_batch, key, self.alpha_state, self.actor_state, self.critic_state, self.critic_target_params)

        log_info['real_batch_rewards'] = real_batch.rewards.sum()
        log_info['real_batch_rewards_min'] = real_batch.rewards.min()
        log_info['real_batch_rewards_max'] = real_batch.rewards.max()
        log_info['real_batch_actions'] = abs(real_batch.actions).reshape(-1).sum()
        log_info['real_batch_observations'] = abs(real_batch.observations).mean(0).sum()
        log_info['real_batch_discounts'] = real_batch.discounts.sum()
        log_info['model_batch_rewards'] = model_batch.rewards.sum()
        log_info['model_batch_rewards_min'] = model_batch.rewards.min()
        log_info['model_batch_rewards_max'] = model_batch.rewards.max()
        log_info['model_batch_actions'] = abs(model_batch.actions).reshape(-1).sum()
        log_info['model_batch_observations'] = abs(model_batch.observations).mean(0).sum()
        log_info['model_batch_discounts'] = model_batch.discounts.sum()
        log_info['model_buffer_size'] = model_buffer.size
        log_info['model_buffer_ptr'] = model_buffer.ptr
        self.update_step += 1
        return log_info

    def save(self, fname: str, cnt: int):
        checkpoints.save_checkpoint(fname, self.actor_state, cnt, prefix="actor_", keep=20, overwrite=True)
        checkpoints.save_checkpoint(fname, self.critic_state, cnt, prefix="critic_", keep=20, overwrite=True)

    def load(self, ckpt_dir, step):
        self.actor_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir,
                                                          target=self.actor_state,
                                                          step=step,
                                                          prefix="actor_")
        self.critic_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir,
                                                           target=self.critic_state,
                                                           step=step,
                                                           prefix="critic_")


# no importance sampling
class COMBOAgent_noimp:
    def __init__(self,
                 env_name: str = "hopper-medium-v2",
                 obs_dim: int = 11,
                 act_dim: int = 3,
                 max_action: float = 1.0,
                 seed: int = 42,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 lr_critic: float = 3e-4,
                 lr_actor: float = 1e-4,
                 target_entropy: float = None,
                 backup_entropy: bool = False,
                 num_random: int = 10,
                 min_q_weight: float = 3.0,

                 # COMBO
                 horizon: int = 5,
                 noise_scale: float = 0.0,
                 real_ratio: float = 0.5,
                 ensemble_num: int = 7,
                 elite_num: int = 5,
                 max_patience: int = 5,
                 batch_size: int = 256,
                 rollout_batch_size: int = 10000,
                 holdout_ratio: float = 0.1,
                 rollout_random: bool = False,
                 hidden_dims: Sequence[int] = (256, 256, 256),
                 initializer: str = 'orthogonal'):
                 

        self.update_step = 0
        self.max_action = max_action
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.backup_entropy = backup_entropy
        if target_entropy is None:
            self.target_entropy = -act_dim
        else:
            self.target_entropy = target_entropy

        # Initialize random keys
        self.rng = jax.random.PRNGKey(seed)
        self.rng, self.rollout_rng, actor_key, critic_key = jax.random.split(self.rng, 4)

        # Dummy inputs
        dummy_obs = jnp.ones([1, obs_dim], dtype=jnp.float32)
        dummy_act = jnp.ones([1, act_dim], dtype=jnp.float32)

        # Initialize the Actor
        self.actor = Actor(act_dim=act_dim, max_action=max_action,
                           hidden_dims=hidden_dims, initializer=initializer)
        actor_params = self.actor.init(actor_key, actor_key, dummy_obs)["params"]
        self.actor_state = train_state.TrainState.create(
            apply_fn=Actor.apply,
            params=actor_params,
            tx=optax.chain(optax.clip(1.0), optax.adam(learning_rate=lr_actor)))

        # Initialize the Critic
        self.critic = DoubleCritic(hidden_dims=hidden_dims, initializer=initializer)
        critic_params = self.critic.init(critic_key, dummy_obs, dummy_act)["params"]
        self.critic_target_params = critic_params
        self.critic_state = train_state.TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=optax.chain(optax.clip(1.0), optax.adam(learning_rate=lr_critic)))

        # Entropy tuning
        self.rng, alpha_key = jax.random.split(self.rng, 2)
        self.log_alpha = Scalar(0.0)
        self.alpha_state = train_state.TrainState.create(
            apply_fn=None,
            params=self.log_alpha.init(alpha_key)["params"],
            tx=optax.chain(optax.clip(1.0), optax.adam(lr_actor))
        )

        # Initialize the Dynamics Model
        self.model = DynamicsModel(env_name=env_name,
                                   seed=seed,
                                   ensemble_num=ensemble_num,   
                                   elite_num=elite_num,
                                   noise_scale=noise_scale)

        # COMBO parameters
        self.horizon = horizon
        self.num_random = num_random
        self.min_q_weight = min_q_weight
        self.rollout_random = rollout_random
        self.batch_size = batch_size
        self.rollout_batch_size = rollout_batch_size

        self.max_patience = max_patience
        self.holdout_ratio = holdout_ratio
        self.ensemble_num = ensemble_num
        self.real_ratio = real_ratio 
        self.real_batch_size = int(real_ratio * batch_size)
        self.model_batch_size = batch_size - self.real_batch_size
        self.real_batch_ratio = self.batch_size / self.real_batch_size
        self.model_batch_ratio = self.batch_size / self.model_batch_size
        self.masks_real = self.real_batch_ratio * np.concatenate([np.ones(self.real_batch_size),
                                                                  np.zeros(self.model_batch_size)])
        # if env_name in ['hopper-medium-v2', 'hopper-medium-replay-v2', 'halfcheetah-medium-v2', "halfcheetah-medium-replay-v2"]:
        if env_name in ['hopper-medium-v2', 'hopper-medium-replay-v2']:
            self.masks_model = self.model_batch_ratio * np.concatenate([np.zeros(self.real_batch_size), np.ones(self.model_batch_size)])
        else:
            self.masks_model = np.ones(self.batch_size)

    @functools.partial(jax.jit, static_argnames=("self"))
    def select_action(self, params: FrozenDict, rng: Any, observation: np.ndarray, eval_mode: bool = False) -> jnp.ndarray:
        rng, sample_rng = jax.random.split(rng)
        mean_action, sampled_action, _ = self.actor.apply({"params": params}, sample_rng, observation)
        return rng, jnp.where(eval_mode, mean_action, sampled_action)

    @functools.partial(jax.jit, static_argnames=("self"), device=jax.devices("cpu")[0])
    def eval_select_action(self, params: FrozenDict, rng: Any, observation: np.ndarray) -> jnp.ndarray:
        mean_action, _, _ = self.actor.apply({"params": params}, rng, observation)
        return rng, mean_action

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self,
                   batch: Batch,
                   key: jnp.ndarray,
                   alpha_state: train_state.TrainState,
                   actor_state: train_state.TrainState,
                   critic_state: train_state.TrainState,
                   critic_target_params: FrozenDict):

        # For use in loss_fn without apply gradients
        frozen_actor_params = actor_state.params
        frozen_critic_params = critic_state.params

        def loss_fn(alpha_params: FrozenDict,
                    actor_params: FrozenDict,
                    critic_params: FrozenDict, 
                    rng: jnp.ndarray,
                    observation: jnp.ndarray,
                    action: jnp.ndarray,
                    reward: jnp.ndarray,
                    next_observation: jnp.ndarray,
                    discount: jnp.ndarray,
                    mask_real: jnp.ndarray,
                    mask_model: jnp.ndarray):
            """compute loss for a single transition"""
            rng, rng1, rng2 = jax.random.split(rng, 3)

            # Sample actions with Actor
            _, sampled_action, logp = self.actor.apply({"params": actor_params}, rng1, observation)

            # Alpha loss: stop gradient to avoid affect Actor parameters
            log_alpha = self.log_alpha.apply({"params": alpha_params})
            alpha_loss = -log_alpha * jax.lax.stop_gradient(logp + self.target_entropy)
            alpha = jnp.exp(log_alpha)
            alpha = jax.lax.stop_gradient(alpha)  # stop gradient to avoid affect Alpha parameters

            # We use frozen_params so that gradients can flow back to the actor without being used to update the critic.
            sampled_q1, sampled_q2 = self.critic.apply({"params": frozen_critic_params}, observation, sampled_action)
            sampled_q = jnp.minimum(sampled_q1, sampled_q2)

            # Actor loss
            actor_loss = alpha * logp - sampled_q

            # Critic loss
            q1, q2 = self.critic.apply({"params": critic_params}, observation, action)

            # Use frozen_actor_params to avoid affect Actor parameters
            _, next_action, logp_next_action = self.actor.apply({"params": frozen_actor_params}, rng2, next_observation)
            next_q1, next_q2 = self.critic.apply({"params": critic_target_params}, next_observation, next_action)
            next_q = jnp.minimum(next_q1, next_q2)
            if self.backup_entropy:
                next_q -= alpha * logp_next_action
            target_q = reward + self.gamma * discount * next_q
            critic_loss1 = 0.5 * (q1 - target_q)**2
            critic_loss2 = 0.5 * (q2 - target_q)**2
            critic_loss = critic_loss1 + critic_loss2

            # COMBO CQL loss
            rng3, rng4 = jax.random.split(rng, 2)
            cql_random_actions = jax.random.uniform(rng3, shape=(self.num_random, self.act_dim), minval=-self.max_action, maxval=self.max_action)
            repeat_observations = jnp.repeat(jnp.expand_dims(observation, axis=0), repeats=self.num_random, axis=0)
            _, cql_sampled_actions, cql_logp = self.actor.apply({"params": frozen_actor_params}, rng4, repeat_observations)
            cql_random_q1, cql_random_q2 = self.critic.apply({"params": critic_params}, repeat_observations, cql_random_actions)
            cql_q1, cql_q2 = self.critic.apply({"params": critic_params}, repeat_observations, cql_sampled_actions)
            cql_concat_q1 = jnp.concatenate([cql_random_q1, cql_q1])
            cql_concat_q2 = jnp.concatenate([cql_random_q2, cql_q2])
            ood_q1 = jax.scipy.special.logsumexp(cql_concat_q1)
            ood_q2 = jax.scipy.special.logsumexp(cql_concat_q2)

            # only use real transition to compute the cql loss
            # compute logsumexp loss w.r.t model_states 
            cql1_loss = (ood_q1*mask_real - q1*mask_real) * self.min_q_weight
            cql2_loss = (ood_q2*mask_real - q2*mask_real) * self.min_q_weight

            total_loss = alpha_loss + actor_loss + critic_loss + cql1_loss + cql2_loss
            log_info = {
                "critic_loss1": critic_loss1,
                "critic_loss2": critic_loss2,
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "alpha_loss": alpha_loss,
                "cql1_loss": cql1_loss,
                "cql2_loss": cql2_loss, 
                "q1": q1,
                "q2": q2,
                "target_q": target_q,
                "sampled_q": sampled_q,
                "ood_q1": ood_q1,
                "ood_q2": ood_q2,
                "cql_q1": cql_q1.mean(),
                "cql_q2": cql_q2.mean(),
                "random_q1": cql_random_q1.mean(),
                "random_q2": cql_random_q2.mean(),
                "alpha": alpha,
                "logp": logp,
                "min_q_weight": self.min_q_weight,
                "logp_next_action": logp_next_action
            }

            return total_loss, log_info

        grad_fn = jax.vmap(
            jax.value_and_grad(loss_fn, argnums=(0, 1, 2), has_aux=True),
            in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, 0, 0))
        keys = jnp.stack(jax.random.split(key, num=batch.actions.shape[0]))
        (_, log_info), gradients = grad_fn(alpha_state.params,
                                           actor_state.params,
                                           critic_state.params,
                                           keys,
                                           batch.observations,
                                           batch.actions,
                                           batch.rewards,
                                           batch.next_observations,
                                           batch.discounts,
                                           self.masks_real,
                                           self.masks_model)
        gradients = jax.tree_map(functools.partial(jnp.mean, axis=0), gradients)
        extra_log_info = {
            'q1_min': log_info['q1'].min(),
            'q1_max': log_info['q1'].max(),
            'q1_std': log_info['q1'].std(),
            'q2_min': log_info['q2'].min(),
            'q2_max': log_info['q2'].max(),
            'q2_std': log_info['q2'].std(),
            'target_q_min': log_info['target_q'].min(),
            'target_q_max': log_info['target_q'].max(),
            'target_q_std': log_info['target_q'].std(),
            'ood_q1_min': log_info['ood_q1'].min(),
            'ood_q1_max': log_info['ood_q1'].max(),
            'ood_q1_std': log_info['ood_q1'].std(),
            'ood_q2_min': log_info['ood_q2'].min(),
            'ood_q2_max': log_info['ood_q2'].max(),
            'ood_q2_std': log_info['ood_q2'].std(),
            'critic_loss_min': log_info['critic_loss'].min(),
            'critic_loss_max': log_info['critic_loss'].max(),
            'critic_loss_std': log_info['critic_loss'].std(),
            'critic_loss1_min': log_info['critic_loss1'].min(),
            'critic_loss1_max': log_info['critic_loss1'].max(),
            'critic_loss1_std': log_info['critic_loss1'].std(),
            'critic_loss2_min': log_info['critic_loss2'].min(),
            'critic_loss2_max': log_info['critic_loss2'].max(),
            'critic_loss2_std': log_info['critic_loss2'].std(),
            'real_critic_loss': log_info['critic_loss'][:self.real_batch_size].mean(),
            'fake_critic_loss': log_info['critic_loss'][self.real_batch_size:].mean(),
            'real_critic_loss_max': log_info['critic_loss'][:self.real_batch_size].max(),
            'fake_critic_loss_min': log_info['critic_loss'][self.real_batch_size:].min(),
            'cql1_loss_min': log_info['cql1_loss'].min(),
            'cql1_loss_max': log_info['cql1_loss'].max(),
            'cql1_loss_std': log_info['cql1_loss'].std(),
            'cql2_loss_min': log_info['cql2_loss'].min(),
            'cql2_loss_max': log_info['cql2_loss'].max(),
            'cql2_loss_std': log_info['cql2_loss'].std(),
        }
        log_info = jax.tree_map(functools.partial(jnp.mean, axis=0), log_info)
        log_info.update(extra_log_info)
        alpha_grads, actor_grads, critic_grads = gradients

        # Update TrainState
        new_alpha_state = alpha_state.apply_gradients(grads=alpha_grads)
        new_actor_state = actor_state.apply_gradients(grads=actor_grads)
        new_critic_state = critic_state.apply_gradients(grads=critic_grads)
        new_critic_target_params = target_update(new_critic_state.params, critic_target_params, self.tau)
        return new_alpha_state, new_actor_state, new_critic_state, new_critic_target_params, log_info

    def update(self, replay_buffer, model_buffer):
        select_action = jax.vmap(self.select_action, in_axes=(None, 0, 0, None))
        if self.update_step % 1000 == 0:
            observations = replay_buffer.sample(self.rollout_batch_size).observations
            sample_rng = jnp.stack(jax.random.split(self.rollout_rng, num=self.rollout_batch_size))
            for t in range(self.horizon):
                self.rollout_rng, rollout_key = jax.random.split(self.rollout_rng, 2)
                if self.rollout_random:
                    actions = np.random.uniform(low=-self.max_action, high=self.max_action,
                                                size=(len(observations), self.act_dim))
                else:
                    sample_rng, actions = select_action(self.actor_state.params, sample_rng, observations, False)

                # normalize states and actions
                normalized_observations = self.model.normalize(observations)
                next_observations, rewards, dones = self.model.step(rollout_key, normalized_observations, actions)
                nonterminal_mask = ~dones
                model_buffer.add_batch(observations, actions, next_observations, rewards, dones)
                if nonterminal_mask.sum() == 0: break
                observations = next_observations[nonterminal_mask]
                sample_rng = sample_rng[nonterminal_mask]

        # sample from real & model buffer
        real_batch = replay_buffer.sample(self.real_batch_size)
        model_batch = model_buffer.sample(self.model_batch_size)
        concat_batch = Batch(
            observations=np.concatenate([real_batch.observations, model_batch.observations], axis=0),
            actions = np.concatenate([real_batch.actions, model_batch.actions], axis=0),
            rewards = np.concatenate([real_batch.rewards, model_batch.rewards], axis=0),
            discounts = np.concatenate([real_batch.discounts, model_batch.discounts], axis=0),
            next_observations = np.concatenate([real_batch.next_observations, model_batch.next_observations], axis=0)
        )

        # CQL training with COMBO
        self.rng, key = jax.random.split(self.rng)
        (self.alpha_state, self.actor_state, self.critic_state, self.critic_target_params, log_info) = self.train_step(
            concat_batch, key, self.alpha_state, self.actor_state, self.critic_state, self.critic_target_params)

        log_info['real_batch_rewards'] = real_batch.rewards.sum()
        log_info['real_batch_rewards_min'] = real_batch.rewards.min()
        log_info['real_batch_rewards_max'] = real_batch.rewards.max()
        log_info['real_batch_actions'] = abs(real_batch.actions).reshape(-1).sum()
        log_info['real_batch_observations'] = abs(real_batch.observations).mean(0).sum()
        log_info['real_batch_discounts'] = real_batch.discounts.sum()
        log_info['model_batch_rewards'] = model_batch.rewards.sum()
        log_info['model_batch_rewards_min'] = model_batch.rewards.min()
        log_info['model_batch_rewards_max'] = model_batch.rewards.max()
        log_info['model_batch_actions'] = abs(model_batch.actions).reshape(-1).sum()
        log_info['model_batch_observations'] = abs(model_batch.observations).mean(0).sum()
        log_info['model_batch_discounts'] = model_batch.discounts.sum()
        log_info['model_buffer_size'] = model_buffer.size
        log_info['model_buffer_ptr'] = model_buffer.ptr
        self.update_step += 1
        return log_info

    def save(self, fname: str, cnt: int):
        checkpoints.save_checkpoint(fname, self.actor_state, cnt, prefix="actor_", keep=20, overwrite=True)
        checkpoints.save_checkpoint(fname, self.critic_state, cnt, prefix="critic_", keep=20, overwrite=True)

    def load(self, ckpt_dir, step):
        self.actor_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir,
                                                          target=self.actor_state,
                                                          step=step,
                                                          prefix="actor_")
        self.critic_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir,
                                                           target=self.critic_state,
                                                           step=step,
                                                           prefix="critic_")


# only use model data to learn the critic
class COMBOAgent_cqlreal:
    def __init__(self,
                 env_name: str = "hopper-medium-v2",
                 obs_dim: int = 11,
                 act_dim: int = 3,
                 max_action: float = 1.0,
                 seed: int = 42,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 lr_critic: float = 3e-4,
                 lr_actor: float = 1e-4,
                 target_entropy: float = None,
                 backup_entropy: bool = False,
                 num_random: int = 10,
                 min_q_weight: float = 3.0,

                 # COMBO
                 horizon: int = 5,
                 noise_scale: float = 0.0,
                 real_ratio: float = 0.5,
                 ensemble_num: int = 7,
                 elite_num: int = 5,
                 max_patience: int = 5,
                 batch_size: int = 256,
                 rollout_batch_size: int = 10000,
                 holdout_ratio: float = 0.1,
                 rollout_random: bool = False,
                 hidden_dims: Sequence[int] = (256, 256, 256),
                 initializer: str = 'orthogonal'):
                 

        self.update_step = 0
        self.max_action = max_action
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.backup_entropy = backup_entropy
        if target_entropy is None:
            self.target_entropy = -act_dim
        else:
            self.target_entropy = target_entropy

        # Initialize random keys
        self.rng = jax.random.PRNGKey(seed)
        self.rng, self.rollout_rng, actor_key, critic_key = jax.random.split(self.rng, 4)

        # Dummy inputs
        dummy_obs = jnp.ones([1, obs_dim], dtype=jnp.float32)
        dummy_act = jnp.ones([1, act_dim], dtype=jnp.float32)

        # Initialize the Actor
        self.actor = Actor(act_dim=act_dim, max_action=max_action,
                           hidden_dims=hidden_dims, initializer=initializer)
        actor_params = self.actor.init(actor_key, actor_key, dummy_obs)["params"]
        self.actor_state = train_state.TrainState.create(
            apply_fn=Actor.apply,
            params=actor_params,
            tx=optax.chain(optax.clip(1.0), optax.adam(learning_rate=lr_actor)))

        # Initialize the Critic
        self.critic = DoubleCritic(hidden_dims=hidden_dims, initializer=initializer)
        critic_params = self.critic.init(critic_key, dummy_obs, dummy_act)["params"]
        self.critic_target_params = critic_params
        self.critic_state = train_state.TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=optax.chain(optax.clip(1.0), optax.adam(learning_rate=lr_critic)))

        # Entropy tuning
        self.rng, alpha_key = jax.random.split(self.rng, 2)
        self.log_alpha = Scalar(0.0)
        self.alpha_state = train_state.TrainState.create(
            apply_fn=None,
            params=self.log_alpha.init(alpha_key)["params"],
            tx=optax.chain(optax.clip(1.0), optax.adam(lr_actor))
        )

        # Initialize the Dynamics Model
        self.model = DynamicsModel(env_name=env_name,
                                   seed=seed,
                                   ensemble_num=ensemble_num,   
                                   elite_num=elite_num,
                                   noise_scale=noise_scale)

        # COMBO parameters
        self.horizon = horizon
        self.num_random = num_random
        self.min_q_weight = min_q_weight
        self.rollout_random = rollout_random
        self.batch_size = batch_size
        self.rollout_batch_size = rollout_batch_size

        self.max_patience = max_patience
        self.holdout_ratio = holdout_ratio
        self.ensemble_num = ensemble_num
        self.real_ratio = real_ratio 
        self.real_batch_size = int(real_ratio * batch_size)
        self.model_batch_size = batch_size - self.real_batch_size
        self.real_batch_ratio = self.batch_size / self.real_batch_size
        self.model_batch_ratio = self.batch_size / self.model_batch_size
        self.masks_real = self.real_batch_ratio * np.concatenate([np.ones(self.real_batch_size),
                                                                  np.zeros(self.model_batch_size)])
        # if env_name in ['hopper-medium-v2', 'hopper-medium-replay-v2', 'halfcheetah-medium-v2', "halfcheetah-medium-replay-v2"]:
        if env_name in ['hopper-medium-v2', 'hopper-medium-replay-v2']:
            self.masks_model = self.model_batch_ratio * np.concatenate([np.zeros(self.real_batch_size), np.ones(self.model_batch_size)])
        else:
            self.masks_model = np.ones(self.batch_size)

    @functools.partial(jax.jit, static_argnames=("self"))
    def select_action(self, params: FrozenDict, rng: Any, observation: np.ndarray, eval_mode: bool = False) -> jnp.ndarray:
        rng, sample_rng = jax.random.split(rng)
        mean_action, sampled_action, _ = self.actor.apply({"params": params}, sample_rng, observation)
        return rng, jnp.where(eval_mode, mean_action, sampled_action)

    @functools.partial(jax.jit, static_argnames=("self"), device=jax.devices("cpu")[0])
    def eval_select_action(self, params: FrozenDict, rng: Any, observation: np.ndarray) -> jnp.ndarray:
        mean_action, _, _ = self.actor.apply({"params": params}, rng, observation)
        return rng, mean_action

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self,
                   batch: Batch,
                   key: jnp.ndarray,
                   alpha_state: train_state.TrainState,
                   actor_state: train_state.TrainState,
                   critic_state: train_state.TrainState,
                   critic_target_params: FrozenDict):

        # For use in loss_fn without apply gradients
        frozen_actor_params = actor_state.params
        frozen_critic_params = critic_state.params

        def loss_fn(alpha_params: FrozenDict,
                    actor_params: FrozenDict,
                    critic_params: FrozenDict, 
                    rng: jnp.ndarray,
                    observation: jnp.ndarray,
                    action: jnp.ndarray,
                    reward: jnp.ndarray,
                    next_observation: jnp.ndarray,
                    discount: jnp.ndarray,
                    mask_real: jnp.ndarray,
                    mask_model: jnp.ndarray):
            """compute loss for a single transition"""
            rng, rng1, rng2 = jax.random.split(rng, 3)

            # Sample actions with Actor
            _, sampled_action, logp = self.actor.apply({"params": actor_params}, rng1, observation)

            # Alpha loss: stop gradient to avoid affect Actor parameters
            log_alpha = self.log_alpha.apply({"params": alpha_params})
            alpha_loss = -log_alpha * jax.lax.stop_gradient(logp + self.target_entropy)
            alpha = jnp.exp(log_alpha)
            alpha = jax.lax.stop_gradient(alpha)  # stop gradient to avoid affect Alpha parameters

            # We use frozen_params so that gradients can flow back to the actor without being used to update the critic.
            sampled_q1, sampled_q2 = self.critic.apply({"params": frozen_critic_params}, observation, sampled_action)
            sampled_q = jnp.minimum(sampled_q1, sampled_q2)

            # Actor loss
            actor_loss = alpha * logp - sampled_q

            # Critic loss
            q1, q2 = self.critic.apply({"params": critic_params}, observation, action)

            # Use frozen_actor_params to avoid affect Actor parameters
            _, next_action, logp_next_action = self.actor.apply({"params": frozen_actor_params}, rng2, next_observation)
            next_q1, next_q2 = self.critic.apply({"params": critic_target_params}, next_observation, next_action)
            next_q = jnp.minimum(next_q1, next_q2)
            if self.backup_entropy:
                next_q -= alpha * logp_next_action
            target_q = reward + self.gamma * discount * next_q
            critic_loss1 = 0.5 * (q1 - target_q)**2
            critic_loss2 = 0.5 * (q2 - target_q)**2
            critic_loss = critic_loss1 + critic_loss2

            # COMBO CQL loss
            rng3, rng4 = jax.random.split(rng, 2)
            cql_random_actions = jax.random.uniform(rng3, shape=(self.num_random, self.act_dim), minval=-self.max_action, maxval=self.max_action)
            repeat_observations = jnp.repeat(jnp.expand_dims(observation, axis=0), repeats=self.num_random, axis=0)
            _, cql_sampled_actions, cql_logp = self.actor.apply({"params": frozen_actor_params}, rng4, repeat_observations)
            cql_random_q1, cql_random_q2 = self.critic.apply({"params": critic_params}, repeat_observations, cql_random_actions)
            cql_q1, cql_q2 = self.critic.apply({"params": critic_params}, repeat_observations, cql_sampled_actions)
            random_density = np.log(0.5 ** self.act_dim)
            cql_concat_q1 = jnp.concatenate([cql_random_q1 - random_density, cql_q1 - cql_logp])
            cql_concat_q2 = jnp.concatenate([cql_random_q2 - random_density, cql_q2 - cql_logp])
            ood_q1 = jax.scipy.special.logsumexp(cql_concat_q1)
            ood_q2 = jax.scipy.special.logsumexp(cql_concat_q2)

            # only use real transition to compute the cql loss
            # compute logsumexp loss w.r.t model_states 
            cql1_loss = (ood_q1*mask_real - q1*mask_real) * self.min_q_weight
            cql2_loss = (ood_q2*mask_real - q2*mask_real) * self.min_q_weight

            total_loss = alpha_loss + actor_loss + critic_loss + cql1_loss + cql2_loss
            log_info = {
                "critic_loss1": critic_loss1,
                "critic_loss2": critic_loss2,
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "alpha_loss": alpha_loss,
                "cql1_loss": cql1_loss,
                "cql2_loss": cql2_loss, 
                "q1": q1,
                "q2": q2,
                "target_q": target_q,
                "sampled_q": sampled_q,
                "ood_q1": ood_q1,
                "ood_q2": ood_q2,
                "cql_q1": cql_q1.mean(),
                "cql_q2": cql_q2.mean(),
                "random_q1": cql_random_q1.mean(),
                "random_q2": cql_random_q2.mean(),
                "alpha": alpha,
                "logp": logp,
                "min_q_weight": self.min_q_weight,
                "logp_next_action": logp_next_action
            }

            return total_loss, log_info

        grad_fn = jax.vmap(
            jax.value_and_grad(loss_fn, argnums=(0, 1, 2), has_aux=True),
            in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, 0, 0))
        keys = jnp.stack(jax.random.split(key, num=batch.actions.shape[0]))
        (_, log_info), gradients = grad_fn(alpha_state.params,
                                           actor_state.params,
                                           critic_state.params,
                                           keys,
                                           batch.observations,
                                           batch.actions,
                                           batch.rewards,
                                           batch.next_observations,
                                           batch.discounts,
                                           self.masks_real,
                                           self.masks_model)
        gradients = jax.tree_map(functools.partial(jnp.mean, axis=0), gradients)
        extra_log_info = {
            'q1_min': log_info['q1'].min(),
            'q1_max': log_info['q1'].max(),
            'q1_std': log_info['q1'].std(),
            'q2_min': log_info['q2'].min(),
            'q2_max': log_info['q2'].max(),
            'q2_std': log_info['q2'].std(),
            'target_q_min': log_info['target_q'].min(),
            'target_q_max': log_info['target_q'].max(),
            'target_q_std': log_info['target_q'].std(),
            'ood_q1_min': log_info['ood_q1'].min(),
            'ood_q1_max': log_info['ood_q1'].max(),
            'ood_q1_std': log_info['ood_q1'].std(),
            'ood_q2_min': log_info['ood_q2'].min(),
            'ood_q2_max': log_info['ood_q2'].max(),
            'ood_q2_std': log_info['ood_q2'].std(),
            'critic_loss_min': log_info['critic_loss'].min(),
            'critic_loss_max': log_info['critic_loss'].max(),
            'critic_loss_std': log_info['critic_loss'].std(),
            'critic_loss1_min': log_info['critic_loss1'].min(),
            'critic_loss1_max': log_info['critic_loss1'].max(),
            'critic_loss1_std': log_info['critic_loss1'].std(),
            'critic_loss2_min': log_info['critic_loss2'].min(),
            'critic_loss2_max': log_info['critic_loss2'].max(),
            'critic_loss2_std': log_info['critic_loss2'].std(),
            'real_critic_loss': log_info['critic_loss'][:self.real_batch_size].mean(),
            'fake_critic_loss': log_info['critic_loss'][self.real_batch_size:].mean(),
            'real_critic_loss_max': log_info['critic_loss'][:self.real_batch_size].max(),
            'fake_critic_loss_min': log_info['critic_loss'][self.real_batch_size:].min(),
            'cql1_loss_min': log_info['cql1_loss'].min(),
            'cql1_loss_max': log_info['cql1_loss'].max(),
            'cql1_loss_std': log_info['cql1_loss'].std(),
            'cql2_loss_min': log_info['cql2_loss'].min(),
            'cql2_loss_max': log_info['cql2_loss'].max(),
            'cql2_loss_std': log_info['cql2_loss'].std(),
        }
        log_info = jax.tree_map(functools.partial(jnp.mean, axis=0), log_info)
        log_info.update(extra_log_info)
        alpha_grads, actor_grads, critic_grads = gradients

        # Update TrainState
        new_alpha_state = alpha_state.apply_gradients(grads=alpha_grads)
        new_actor_state = actor_state.apply_gradients(grads=actor_grads)
        new_critic_state = critic_state.apply_gradients(grads=critic_grads)
        new_critic_target_params = target_update(new_critic_state.params, critic_target_params, self.tau)
        return new_alpha_state, new_actor_state, new_critic_state, new_critic_target_params, log_info

    def update(self, replay_buffer, model_buffer):
        select_action = jax.vmap(self.select_action, in_axes=(None, 0, 0, None))
        if self.update_step % 1000 == 0:
            observations = replay_buffer.sample(self.rollout_batch_size).observations
            sample_rng = jnp.stack(jax.random.split(self.rollout_rng, num=self.rollout_batch_size))
            for t in range(self.horizon):
                self.rollout_rng, rollout_key = jax.random.split(self.rollout_rng, 2)
                if self.rollout_random:
                    actions = np.random.uniform(low=-self.max_action, high=self.max_action,
                                                size=(len(observations), self.act_dim))
                else:
                    sample_rng, actions = select_action(self.actor_state.params, sample_rng, observations, False)

                # normalize states and actions
                normalized_observations = self.model.normalize(observations)
                next_observations, rewards, dones = self.model.step(rollout_key, normalized_observations, actions)
                nonterminal_mask = ~dones
                model_buffer.add_batch(observations, actions, next_observations, rewards, dones)
                if nonterminal_mask.sum() == 0: break
                observations = next_observations[nonterminal_mask]
                sample_rng = sample_rng[nonterminal_mask]

        # sample from real & model buffer
        real_batch = replay_buffer.sample(self.real_batch_size)
        model_batch = model_buffer.sample(self.model_batch_size)
        concat_batch = Batch(
            observations=np.concatenate([real_batch.observations, model_batch.observations], axis=0),
            actions = np.concatenate([real_batch.actions, model_batch.actions], axis=0),
            rewards = np.concatenate([real_batch.rewards, model_batch.rewards], axis=0),
            discounts = np.concatenate([real_batch.discounts, model_batch.discounts], axis=0),
            next_observations = np.concatenate([real_batch.next_observations, model_batch.next_observations], axis=0)
        )

        # CQL training with COMBO
        self.rng, key = jax.random.split(self.rng)
        (self.alpha_state, self.actor_state, self.critic_state, self.critic_target_params, log_info) = self.train_step(
            concat_batch, key, self.alpha_state, self.actor_state, self.critic_state, self.critic_target_params)

        log_info['real_batch_rewards'] = real_batch.rewards.sum()
        log_info['real_batch_rewards_min'] = real_batch.rewards.min()
        log_info['real_batch_rewards_max'] = real_batch.rewards.max()
        log_info['real_batch_actions'] = abs(real_batch.actions).reshape(-1).sum()
        log_info['real_batch_observations'] = abs(real_batch.observations).mean(0).sum()
        log_info['real_batch_discounts'] = real_batch.discounts.sum()
        log_info['model_batch_rewards'] = model_batch.rewards.sum()
        log_info['model_batch_rewards_min'] = model_batch.rewards.min()
        log_info['model_batch_rewards_max'] = model_batch.rewards.max()
        log_info['model_batch_actions'] = abs(model_batch.actions).reshape(-1).sum()
        log_info['model_batch_observations'] = abs(model_batch.observations).mean(0).sum()
        log_info['model_batch_discounts'] = model_batch.discounts.sum()
        log_info['model_buffer_size'] = model_buffer.size
        log_info['model_buffer_ptr'] = model_buffer.ptr
        self.update_step += 1
        return log_info

    def save(self, fname: str, cnt: int):
        checkpoints.save_checkpoint(fname, self.actor_state, cnt, prefix="actor_", keep=20, overwrite=True)
        checkpoints.save_checkpoint(fname, self.critic_state, cnt, prefix="critic_", keep=20, overwrite=True)

    def load(self, ckpt_dir, step):
        self.actor_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir,
                                                          target=self.actor_state,
                                                          step=step,
                                                          prefix="actor_")
        self.critic_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir,
                                                           target=self.critic_state,
                                                           step=step,
                                                           prefix="critic_")


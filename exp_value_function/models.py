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
from flax import serialization
from flax.core import FrozenDict
from flax.training import checkpoints, train_state


# Dynamics Model
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
                 model_dir: str = "./dynamics_models"):

        # Model parameters
        self.seed = seed
        self.lr = lr
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

    def rollout(self, params, observations, actions):

        @jax.jit
        def rollout_fn(observation, action):
            x = jnp.concatenate([observation, action], axis=-1).reshape(1, -1)
            model_mu, _ = self.model.apply({"params": params}, x)
            observation_mu, _ = jnp.split(model_mu, [self.obs_dim], axis=-1)
            uncertainty = jnp.linalg.norm(observation_mu -
                                          observation_mu.mean(0),
                                          axis=-1).max()
            return uncertainty

        uncertainties = jax.vmap(rollout_fn, in_axes=(0, 0))(observations,
                                                             actions)
        return uncertainties

    def step(self, observations, actions):
        normalized_observations = self.normalize(observations)
        model_stds = self.rollout(self.model_state.params,
                                  normalized_observations, actions)
        return model_stds

    def normalize(self, observations):
        new_observations = (observations - self.obs_mean) / self.obs_std
        return new_observations

    def denormalize(self, observations):
        new_observations = observations * self.obs_std + self.obs_mean
        return new_observations

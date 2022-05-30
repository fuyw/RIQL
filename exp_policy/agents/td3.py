import functools
from typing import Callable, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax import serialization
from flax.core import FrozenDict
from flax.training import checkpoints, train_state


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


class Critic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
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
    hidden_dims: Sequence[int] = (256, 256)
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
    
    def encode(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        embeddings = self.critic1.encode(observations, actions)
        return embeddings


class Actor(nn.Module):
    act_dim: int
    max_action: float = 1.0
    hidden_dims: Sequence[int] = (256, 256)
    initializer: str = "orthogonal"

    def setup(self):
        self.net = MLP(self.hidden_dims, init_fn=init_fn(self.initializer), activate_final=True)
        self.out_layer = nn.Dense(self.act_dim, kernel_init=init_fn(self.initializer, 5/3))  # default gain for `tanh` activation

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        x = self.net(observations)
        x = self.out_layer(x)
        mean_action = nn.tanh(x) * self.max_action
        return mean_action

    def encode(self, observations):
        embeddings = self.net(observations)
        return embeddings


class TD3Agent:
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 max_action: float = 1.0,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 noise_clip: float = 0.5,
                 policy_noise: float = 0.2,
                 policy_freq: int = 2,
                 lr: float = 3e-4,
                 alpha: float = 2.5,
                 seed: int = 42,
                 hidden_dims: Sequence[int] = (256, 256),
                 initializer: str = "glorot_uniform"):

        self.max_action = max_action
        self.noise_clip = noise_clip
        self.policy_noise = policy_noise
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau

        rng = jax.random.PRNGKey(seed)
        self.actor_rng, self.critic_rng = jax.random.split(rng, 2)

        # Dummy inputs
        dummy_obs = jnp.ones([1, obs_dim], dtype=jnp.float32)
        dummy_act = jnp.ones([1, act_dim], dtype=jnp.float32)

        self.actor = Actor(act_dim=act_dim, max_action=max_action,
                           hidden_dims=hidden_dims, initializer=initializer)
        actor_params = self.actor.init(self.actor_rng, dummy_obs)["params"]
        self.actor_target_params = actor_params
        self.actor_state = train_state.TrainState.create(apply_fn=Actor.apply,
                                                         params=actor_params,
                                                         tx=optax.adam(learning_rate=lr))

        # Initialize the critic
        self.critic = DoubleCritic(hidden_dims=hidden_dims, initializer=initializer)
        critic_params = self.critic.init(self.critic_rng, dummy_obs, dummy_act)["params"]
        self.critic_target_params = critic_params
        self.critic_state = train_state.TrainState.create(apply_fn=Critic.apply,
                                                          params=critic_params,
                                                          tx=optax.adam(learning_rate=lr))
        self.update_step = 0

    # sample on cpu can accelerate a little bit
    @functools.partial(jax.jit, static_argnames=("self"), device=jax.devices("cpu")[0])
    def sample_action(self, params: FrozenDict, observation: jnp.ndarray) -> jnp.ndarray:
        sampled_action = self.actor.apply({"params": params}, observation)
        return sampled_action

    def load(self, fname, step):
        self.actor_state = checkpoints.restore_checkpoint(ckpt_dir=fname, target=self.actor_state, step=step, prefix="actor_")
        self.critic_state = checkpoints.restore_checkpoint(ckpt_dir=fname, target=self.critic_state, step=step, prefix="critic_")

    def encode_sa(self, observations, actions):
        embeddings = self.critic.apply({"params": self.critic_state.params},
                                       observations, actions,
                                       method=self.critic.encode)
        return embeddings

    def encode_s(self, observations):
        embeddings = self.actor.apply({"params": self.actor_state.params},
                                      observations, method=self.actor.encode)
        return embeddings

    def Q1(self, observations, actions):
        Qs = self.critic.apply({"params": self.critic_state.params},
                               observations, actions, method=self.critic.Q1)
        return Qs

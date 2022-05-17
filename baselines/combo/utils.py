import collections
import jax
import logging
import numpy as np
from flax.core import FrozenDict


Batch = collections.namedtuple(
    "Batch",
    ["observations", "actions", "rewards", "discounts", "next_observations"])


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, max_size: int = int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.observations = np.zeros((max_size, obs_dim))
        self.actions = np.zeros((max_size, act_dim))
        self.next_observations = np.zeros((max_size, obs_dim))
        self.rewards = np.zeros(max_size)
        self.discounts = np.zeros(max_size)

    def add(self, observation: np.ndarray, action: np.ndarray,
            next_observation: np.ndarray, reward: float, done: float):
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.next_observations[self.ptr] = next_observation
        self.rewards[self.ptr] = reward
        self.discounts[self.ptr] = 1 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, observations, actions, next_observations, rewards, dones):
        add_num = len(actions)
        add_idx = np.arange(self.ptr, self.ptr + add_num) % self.max_size
        self.observations[add_idx] = observations
        self.actions[add_idx] = actions
        self.next_observations[add_idx] = next_observations
        self.rewards[add_idx] = rewards.squeeze()
        self.discounts[add_idx] = 1 - dones.squeeze()
        self.ptr = (self.ptr + add_num) % self.max_size
        self.size = min(self.size + add_num, self.max_size)

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = Batch(observations=self.observations[idx],
                      actions=self.actions[idx],
                      rewards=self.rewards[idx],
                      discounts=self.discounts[idx],
                      next_observations=self.next_observations[idx])
        return batch

    def convert_D4RL(self, dataset):
        self.observations = dataset["observations"]
        self.actions = dataset["actions"]
        self.next_observations = dataset["next_observations"]
        self.rewards = dataset["rewards"].squeeze()
        self.discounts = 1. - dataset["terminals"].squeeze()
        self.size = self.observations.shape[0]

    def normalize_obs(self, eps: float = 1e-3):
        mean = self.observations.mean(0)
        std = self.observations.std(0) + eps
        self.observations = (self.observations - mean)/std
        self.next_observations = (self.next_observations - mean)/std
        return mean, std


def get_logger(fname: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=fname,
                        filemode='w',
                        force=True)
    logger = logging.getLogger()
    return logger


def target_update(params: FrozenDict, target_params: FrozenDict, tau: float) -> FrozenDict:
    def _update(param: FrozenDict, target_param: FrozenDict):
        return tau*param + (1-tau)*target_param
    updated_params = jax.tree_multimap(_update, params, target_params)
    return updated_params


def get_training_data(replay_buffer, ensemble_num=7, holdout_ratio=0.1, eps=1e-3):
    # load the offline data
    observations = replay_buffer.observations
    actions = replay_buffer.actions
    next_observations = replay_buffer.next_observations
    rewards = replay_buffer.rewards.reshape(-1, 1)  # reshape for concatenate
    holdout_num = int(holdout_ratio * len(observations))

    # validation dataset
    permutation = np.random.permutation(len(observations))
    train_idx, target_idx = permutation[holdout_num:], permutation[:holdout_num]

    # split validation set
    train_observations = observations[train_idx]

    # compute the normalize stats
    obs_mean = train_observations.mean(0)
    obs_std = train_observations.std(0) + eps

    # normlaize the data
    observations = (observations - obs_mean) / obs_std
    next_observations = (next_observations - obs_mean) / obs_std
    delta_observations = next_observations - observations

    # prepare for model inputs & outputs
    inputs = np.concatenate([observations, actions], axis=-1)
    targets = np.concatenate([delta_observations, rewards], axis=-1)

    # split the dataset
    inputs, holdout_inputs = inputs[train_idx], inputs[target_idx]
    targets, holdout_targets = targets[train_idx], targets[target_idx]
    holdout_inputs = np.tile(holdout_inputs[None], [ensemble_num, 1, 1])
    holdout_targets = np.tile(holdout_targets[None], [ensemble_num, 1, 1])

    return inputs, targets, holdout_inputs, holdout_targets, obs_mean, obs_std

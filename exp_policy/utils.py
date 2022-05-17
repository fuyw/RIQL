import collections

import jax
import jax.numpy as jnp
import numpy as np

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
        self.observations = (self.observations - mean) / std
        self.next_observations = (self.next_observations - mean) / std
        return mean, std


def load_data(env_name):
    data = np.load(f"saved_buffers/{env_name.split('-')[0]}-v2/L100K.npz")
    observations = data["observations"]
    actions = data["actions"]
    next_observations = data["next_observations"]
    rewards = data["rewards"]
    return observations, actions, rewards, next_observations


def get_kernel_norm(kernel_params: jnp.array):
    return jnp.linalg.norm(kernel_params).item()


def get_sa_embeddings(agent, observations, actions):
    L = len(observations)
    batch_size = 10000
    batch_num = int(np.ceil(L / batch_size))
    encode = jax.jit(agent.encode_sa)
    embeddings = []
    for i in range(batch_num):
        batch_observations = observations[i * batch_size:(i + 1) * batch_size]
        batch_actions = actions[i * batch_size:(i + 1) * batch_size]
        batch_embedding = encode(batch_observations, batch_actions)
        embeddings.append(batch_embedding)
    embeddings = np.concatenate(embeddings,
                                axis=0) + 1e-6  # avoid zero-division
    assert len(embeddings) == L
    return embeddings


def get_s_embeddings(agent, observations):
    L = len(observations)
    batch_size = 10000
    batch_num = int(np.ceil(L / batch_size))
    encode = jax.jit(agent.encode_s)
    embeddings = []
    for i in range(batch_num):
        batch_observations = observations[i * batch_size:(i + 1) * batch_size]
        batch_embedding = encode(batch_observations)
        embeddings.append(batch_embedding)
    embeddings = np.concatenate(embeddings, axis=0) + 1e-6
    assert len(embeddings) == L
    return embeddings


def get_sa_srank(agent, observations, actions):
    # get embeddings
    sa_embeddings = get_sa_embeddings(agent, observations, actions)  # (N, 256)
    sa_embeddings_norm = np.linalg.norm(sa_embeddings, axis=-1, keepdims=True)

    normalized_sa_embeddings = sa_embeddings / sa_embeddings_norm  # avoid zero-division
    normalized_mean_sa_embedding = normalized_sa_embeddings.mean(axis=0)

    # approximate covariance matrix with samples
    sa_cov = np.zeros(shape=(256, 256))
    L = len(observations)
    for tmp_idx in range(L // 5000):
        tmp_embeddings = normalized_sa_embeddings[tmp_idx * 5000:(
            tmp_idx + 1) * 5000] - normalized_mean_sa_embedding
        tmp_sa_cov = np.matmul(tmp_embeddings.reshape(-1, 256, 1),
                               tmp_embeddings.reshape(-1, 1, 256)).sum(0)
        sa_cov += tmp_sa_cov
    sa_cov /= L
    _, sigma_sa, _ = np.linalg.svd(sa_cov, full_matrices=False)

    # compute the effective rank
    cumsum_sigma_sa = sigma_sa.cumsum()
    threshold = cumsum_sigma_sa[-1] * 0.99
    sa_srank = sum(cumsum_sigma_sa <= threshold)
    return {
        "sa_srank": sa_srank,
        "sa_embeddings_norm": sa_embeddings_norm.mean(),
        "sa_embeddings_norm_std": sa_embeddings_norm.std(),
        "sa_embeddings_norm_min": sa_embeddings_norm.min(),
        "sa_embeddings_norm_max": sa_embeddings_norm.max()
    }


def get_sa_effective_dim(agent, observations, actions):
    # get embeddings
    sa_embeddings = get_sa_embeddings(agent, observations, actions)  # (N, 256)
    num_rows, _ = sa_embeddings.shape

    # origin effective dim
    u, s, _ = np.linalg.svd(sa_embeddings,
                            full_matrices=False,
                            compute_uv=True)
    rank = max(np.sum(s >= 1e-5), 1)
    u1 = u[:, :rank]
    projected_basis = np.matmul(u1, np.transpose(u1))
    norms = np.linalg.norm(projected_basis, axis=0, ord=2)**2
    eff_dim1 = num_rows * np.max(norms)

    # normalized effective dim
    sa_embeddings_norm = np.linalg.norm(sa_embeddings, axis=-1, keepdims=True)
    normalized_sa_embeddings = sa_embeddings / sa_embeddings_norm  # avoid zero-division
    u, s, _ = np.linalg.svd(normalized_sa_embeddings,
                            full_matrices=False,
                            compute_uv=True)
    cumsum_s = s.cumsum()
    threshold = cumsum_s[-1] * 0.99
    rank = sum(cumsum_s <= threshold)
    u1 = u[:, :rank]
    projected_basis = np.matmul(u1, np.transpose(u1))
    norms = np.linalg.norm(projected_basis, axis=0, ord=2)**2
    eff_dim2 = num_rows * np.max(norms)

    return {"sa_eff_dim1": eff_dim1, "sa_eff_dim2": eff_dim2}


def get_s_srank(agent, observations):
    # get embeddings
    s_embeddings = get_s_embeddings(agent, observations)  # (N, 256)
    s_embeddings_norm = np.linalg.norm(s_embeddings, axis=-1, keepdims=True)

    normalized_s_embeddings = s_embeddings / s_embeddings_norm  # avoid zero-division
    normalized_mean_s_embedding = normalized_s_embeddings.mean(axis=0)

    # approximate covariance matrix with samples
    s_cov = np.zeros(shape=(256, 256))
    L = len(observations)
    for tmp_idx in range(L // 5000):
        tmp_embeddings = normalized_s_embeddings[tmp_idx * 5000:(
            tmp_idx + 1) * 5000] - normalized_mean_s_embedding
        tmp_s_cov = np.matmul(tmp_embeddings.reshape(-1, 256, 1),
                              tmp_embeddings.reshape(-1, 1, 256)).sum(0)
        s_cov += tmp_s_cov
    s_cov /= L
    _, sigma_s, _ = np.linalg.svd(s_cov, full_matrices=False)

    # compute the effective rank
    cumsum_sigma_s = sigma_s.cumsum()
    threshold = cumsum_sigma_s[-1] * 0.99
    s_srank = sum(cumsum_sigma_s <= threshold)

    return {
        "s_srank": s_srank,
        "s_embeddings_norm": s_embeddings_norm.mean(),
        "s_embeddings_norm_std": s_embeddings_norm.std(),
        "s_embeddings_norm_min": s_embeddings_norm.min(),
        "s_embeddings_norm_max": s_embeddings_norm.max()
    }


def get_s_effective_dim(agent, observations):
    # get embeddings
    s_embeddings = get_s_embeddings(agent, observations)  # (N, 256)
    num_rows, _ = s_embeddings.shape

    # origin effective dim
    u, s, _ = np.linalg.svd(s_embeddings, full_matrices=False, compute_uv=True)
    rank = max(np.sum(s >= 1e-5), 1)
    u1 = u[:, :rank]
    projected_basis = np.matmul(u1, np.transpose(u1))
    norms = np.linalg.norm(projected_basis, axis=0, ord=2)**2
    eff_dim1 = num_rows * np.max(norms)

    # normalized effective dim
    s_embeddings_norm = np.linalg.norm(s_embeddings, axis=-1, keepdims=True)
    normalized_s_embeddings = s_embeddings / s_embeddings_norm  # avoid zero-division
    u, s, _ = np.linalg.svd(normalized_s_embeddings,
                            full_matrices=False,
                            compute_uv=True)
    cumsum_s = s.cumsum()
    threshold = cumsum_s[-1] * 0.99
    rank = sum(cumsum_s <= threshold)
    u1 = u[:, :rank]
    projected_basis = np.matmul(u1, np.transpose(u1))
    norms = np.linalg.norm(projected_basis, axis=0, ord=2)**2
    eff_dim2 = num_rows * np.max(norms)
    return {"s_eff_dim1": eff_dim1, "s_eff_dim2": eff_dim2}


def get_q_value(agent, observations, actions):
    L = len(observations)
    batch_size = 10000
    batch_num = int(np.ceil(L / batch_size))
    Q1 = jax.jit(agent.Q1)
    optimal_Qs = []
    for i in range(batch_num):
        batch_observations = observations[i * batch_size:(i + 1) * batch_size]
        batch_actions = actions[i * batch_size:(i + 1) * batch_size]
        batch_Qs = Q1(batch_observations, batch_actions)
        optimal_Qs.append(batch_Qs)
    optimal_Qs = np.concatenate(optimal_Qs, axis=0)
    assert len(optimal_Qs) == L
    return optimal_Qs


def get_optimal_actions(agent, observations):
    L = len(observations)
    batch_size = 10000
    batch_num = int(np.ceil(L / batch_size))
    optimal_actions = []
    for i in range(batch_num):
        batch_observations = observations[i * batch_size:(i + 1) * batch_size]
        actions = agent.sample_action(agent.actor_state.params,
                                      batch_observations)
        optimal_actions.append(actions)
    optimal_actions = np.concatenate(optimal_actions, axis=0)
    assert len(optimal_actions) == L
    return optimal_actions


def get_dot_product(agent, observations, actions, next_observations):
    L = len(observations)
    batch_size = 10000
    batch_num = int(np.ceil(L / batch_size))
    encode = jax.jit(agent.encode_sa)
    encode_s = jax.jit(agent.encode_s)

    sa_embeddings = []
    next_sa_embeddings = []
    s_embeddings = []
    next_s_embeddings = []

    for i in range(batch_num):
        batch_observations = observations[i * batch_size:(i + 1) * batch_size]
        batch_actions = actions[i * batch_size:(i + 1) * batch_size]
        batch_next_observations = next_observations[i * batch_size:(i + 1) *
                                                    batch_size]
        batch_next_actions = agent.sample_action(agent.actor_state.params,
                                                 batch_next_observations)

        batch_sa_embedding = encode(batch_observations, batch_actions)
        batch_next_sa_embedding = encode(batch_next_observations,
                                         batch_next_actions)
        batch_s_embedding = encode_s(batch_observations)
        batch_next_s_embedding = encode_s(batch_next_observations)

        sa_embeddings.append(batch_sa_embedding)
        next_sa_embeddings.append(batch_next_sa_embedding)
        s_embeddings.append(batch_s_embedding)
        next_s_embeddings.append(batch_next_s_embedding)

    sa_embeddings = np.concatenate(sa_embeddings, axis=0)  # (N, 256)
    next_sa_embeddings = np.concatenate(next_sa_embeddings, axis=0)  # (N, 256)
    s_embeddings = np.concatenate(s_embeddings, axis=0)  # (N, 256)
    next_s_embeddings = np.concatenate(next_s_embeddings, axis=0)  # (N, 256)

    sa_embeddings_norm = np.linalg.norm(sa_embeddings, axis=-1, keepdims=True)
    normalized_sa_embeddings = sa_embeddings / sa_embeddings_norm
    next_sa_embeddings_norm = np.linalg.norm(next_sa_embeddings,
                                             axis=-1,
                                             keepdims=True)
    normalized_next_sa_embeddings = next_sa_embeddings / next_sa_embeddings_norm

    s_embeddings_norm = np.linalg.norm(s_embeddings, axis=-1, keepdims=True)
    normalized_s_embeddings = s_embeddings / s_embeddings_norm
    next_s_embeddings_norm = np.linalg.norm(next_s_embeddings,
                                            axis=-1,
                                            keepdims=True)
    normalized_next_s_embeddings = next_s_embeddings / next_s_embeddings_norm

    sa_dot_product = (sa_embeddings * next_sa_embeddings).sum(axis=-1)
    sa_cosine_similarity = (normalized_sa_embeddings *
                            normalized_next_sa_embeddings).sum(axis=-1)
    s_dot_product = (s_embeddings * next_s_embeddings).sum(axis=-1)
    s_cosine_similarity = (normalized_s_embeddings *
                           normalized_next_s_embeddings).sum(axis=-1)

    return {
        "sa_dot_product": sa_dot_product.mean(),
        "sa_cosine_similarity": sa_cosine_similarity.mean(),
        "s_dot_product": s_dot_product.mean(),
        "s_cosine_similarity": s_cosine_similarity.mean()
    }

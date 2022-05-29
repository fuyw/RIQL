import ml_collections
import numpy as np


def get_config():
    config = ml_collections.ConfigDict()
    config.env_name = "halfcheetah-medium-expert-v2"
    config.log_dir = "logs"
    config.model_dir = "saved_models"
    config.initializer = "glorut_uniform"
    config.hidden_dims = (256, 256, 256)
    config.lr_critic = 3e-4
    config.lr_actor = 1e-4
    config.seed = 0
    config.tau = 0.005
    config.gamma = 0.99
    config.batch_size = 256
    config.num_random = 10
    config.min_q_weight = 5.0
    config.target_entropy = None
    config.backup_entropy = False
    config.with_lagrange = False
    config.lagrange_thresh = 5.0
    config.cql_clip_diff_min = -200 
    config.cql_clip_diff_max = np.inf
    config.eval_freq = 5000
    config.eval_episodes = 10
    config.bc_timesteps = 0
    config.max_target_backup = False
    config.max_timesteps = 1000000
    return config

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.env_name = "hopper-medium-v2"
    config.algo = "combo"
    config.log_dir = "logs"
    config.model_dir = "saved_models"
    config.dynamics_model_dir="saved_dynamics_models"
    config.hidden_dims = (256, 256, 256)
    config.lr_critic = 3e-4
    config.lr_actor = 1e-4
    config.seed = 0
    config.tau = 0.005
    config.gamma = 0.99
    config.batch_size = 256
    config.eval_freq = 5000
    config.eval_episodes = 10
    config.max_timesteps = 1000000
    config.num_random = 10
    config.min_q_weight = 1.0
    config.target_entropy = None
    config.backup_entropy = False
    config.num_random = 10
    config.batch_size = 256
    config.real_ratio = 0.5
    config.horizon = 5
    config.noise_scale = 0.0
    config.rollout_batch_size = 10000
    config.holdout_ratio = 0.01
    config.rollout_random = False
    config.initializer="orthogonal"
    return config

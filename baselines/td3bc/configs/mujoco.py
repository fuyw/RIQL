import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.env_name = "hopper-medium-v2"
    config.algo = "td3bc"
    config.log_dir = "logs"
    config.model_dir = "saved_models"
    config.initializer = "glorot_uniform"
    config.hidden_dims = (256, 256)
    config.lr = 3e-4
    config.seed = 1
    config.tau = 0.005
    config.gamma = 0.99
    config.alpha = 2.5
    config.batch_size = 256
    config.eval_freq = 5000
    config.eval_episodes = 10
    config.max_timesteps = 1000000
    config.expl_noise = 0.1
    config.policy_noise = 0.2
    config.noise_clip = 0.5
    config.policy_freq = 2
    return config

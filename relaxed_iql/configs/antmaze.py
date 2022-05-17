import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.env_name = "antmaze-medium-play-v0"
    config.log_dir = "logs"
    config.model_dir = "saved_models"
    config.initializer = "orthogonal"
    config.hidden_dims = (256, 256)
    config.lr = 3e-4
    config.seed = 0
    config.tau = 0.005
    config.gamma = 0.99
    config.expectile = 0.9
    config.temperature = 10.0
    config.batch_size = 256
    config.eval_freq = 50000
    config.eval_episodes = 100
    config.max_timesteps = 1000000
    return config

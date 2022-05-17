from absl import app, flags
from ml_collections import config_flags
import os
import train


config_flags.DEFINE_config_file("config", default="configs/mujoco.py")
FLAGS = flags.FLAGS


conf_dict = {
    "walker2d-medium-v2":           {"lr_actor": 1e-5, "lr_critic": 1e-4, "min_q_weight": 3.0, "horizon": 1, "rollout_random": False},
    "walker2d-medium-replay-v2":    {"lr_actor": 1e-5, "lr_critic": 1e-4, "min_q_weight": 1.0, "horizon": 1, "rollout_random": False},
    "walker2d-medium-expert-v2":    {"lr_actor": 1e-5, "lr_critic": 1e-4, "min_q_weight": 3.0, "horizon": 1, "rollout_random": False},
    "hopper-medium-v2":             {"lr_actor": 1e-5, "lr_critic": 1e-4, "min_q_weight": 3.0, "horizon": 5, "rollout_random": False},
    "hopper-medium-replay-v2":      {"lr_actor": 1e-4, "lr_critic": 3e-4, "min_q_weight": 1.0, "horizon": 5, "rollout_random": True},
    "hopper-medium-expert-v2":      {"lr_actor": 1e-5, "lr_critic": 1e-4, "min_q_weight": 3.0, "horizon": 3, "rollout_random": False},
    "halfcheetah-medium-v2":        {"lr_actor": 1e-5, "lr_critic": 1e-4, "min_q_weight": 1.0, "horizon": 5, "rollout_random": False},
    "halfcheetah-medium-replay-v2": {"lr_actor": 1e-4, "lr_critic": 3e-4, "min_q_weight": 1.0, "horizon": 5, "rollout_random": False},
    "halfcheetah-medium-expert-v2": {"lr_actor": 1e-5, "lr_critic": 1e-4, "min_q_weight": 5.0, "horizon": 5, "rollout_random": False},
}



def main(argv):
    configs = FLAGS.config
    os.makedirs(f"{configs.log_dir}/{configs.env_name}", exist_ok=True)
    os.makedirs(f"{configs.model_dir}/{configs.env_name}", exist_ok=True)
    if configs.env_name in conf_dict:
        configs.horizon = conf_dict[configs.env_name]["horizon"]
        configs.min_q_weight = conf_dict[configs.env_name]["min_q_weight"]
        configs.rollout_random = conf_dict[configs.env_name]["rollout_random"]
        configs.lr_actor = conf_dict[configs.env_name]["lr_actor"]
        configs.lr_critic = conf_dict[configs.env_name]["lr_critic"]
    train.train_and_evaluate(configs)


if __name__ == '__main__':
    app.run(main)

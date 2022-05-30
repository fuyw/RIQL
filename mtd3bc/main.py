from absl import app, flags
from ml_collections import config_flags
import os
import train


config_flags.DEFINE_config_file("config", default="configs/mujoco.py")
FLAGS = flags.FLAGS


def main(argv):
    configs = FLAGS.config
    if "walker2d" in configs.env_name: configs.horizon = 1
    os.makedirs(f"{configs.log_dir}/{configs.env_name}/{configs.algo}", exist_ok=True)
    os.makedirs(f"{configs.model_dir}/{configs.env_name}/{configs.algo}", exist_ok=True)
    train.train_and_evaluate(configs)


if __name__ == '__main__':
    app.run(main)


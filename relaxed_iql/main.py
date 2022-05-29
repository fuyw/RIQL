from absl import app, flags
from ml_collections import config_flags
import os
import train


config_flags.DEFINE_config_file("config", default="configs/mujoco.py")
FLAGS = flags.FLAGS


mle_alphas = {
    "halfcheetah-medium-v2": 0.1,
    "halfcheetah-medium-replay-v2": 0.25,
    "halfcheetah-medium-expert-v2": 1.0,
    "hopper-medium-v2": 1.0,
    "hopper-medium-replay-v2": 0.25,
    "hopper-medium-expert-v2": 1.0,
    "walker2d-medium-v2": 1.0,
    "walker2d-medium-replay-v2": 0.25,
    "walker2d-medium-expert-v2": 1.5,
}



def main(argv):
    configs = FLAGS.config
    configs.mle_alpha = mle_alphas[configs.env_name]
    os.makedirs(f"{configs.log_dir}/{configs.env_name}", exist_ok=True)
    os.makedirs(f"{configs.model_dir}/{configs.env_name}", exist_ok=True)
    train.train_and_evaluate(configs)


if __name__ == '__main__':
    app.run(main)

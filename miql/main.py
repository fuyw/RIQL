import os

from absl import app, flags
from ml_collections import config_flags

import train

config_flags.DEFINE_config_file("config", default="configs/mujoco.py")
FLAGS = flags.FLAGS

batch_configs = {
    "halfcheetah-medium-v2": "concat_concat_concat",
    "halfcheetah-medium-replay-v2": "concat_concat_concat",
    "halfcheetah-medium-expert-v2": "concat_concat_concat",

    "hopper-medium-v2": "concat_concat_concat",
    "hopper-medium-replay-v2": "real_real_concat",
    "hopper-medium-expert-v2": "real_real_concat",

    "walker2d-medium-v2": "real_concat_concat",
    "walker2d-medium-replay-v2": "real_concat_concat",
    "walker2d-medium-expert-v2": "real_concat_concat",
}

def main(argv):
    configs = FLAGS.config
    if "walker2d" in configs.env_name: configs.horizon = 1
    if configs.algo == "cda":
        configs.batchs = batch_configs[configs.env_name]
    os.makedirs(f"{configs.log_dir}/{configs.env_name}/{configs.algo}", exist_ok=True)
    os.makedirs(f"{configs.model_dir}/{configs.env_name}", exist_ok=True)
    train.train_and_evaluate(configs)


if __name__ == '__main__':
    app.run(main)

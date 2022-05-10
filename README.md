# JaxIQL

A Jax implementation of in-sample Q-learning [(IQL)](https://arxiv.org/abs/2110.06169).

![res](imgs/mujoco.png)

![res](imgs/antmaze.png)

To run experiments in `mujoco` environment
```
python main.py --config=configs/mujoco.py --config.env_name=halfcheetah-medium-v2 --config.seed=0
```

To run experiments in `antmaze` environment
```
python main.py --config=configs/antmaze.py --config.env_name=antmaze-medium-play-v0 --config.seed=0
```
# RIQL

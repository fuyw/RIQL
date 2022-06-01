# A Closer Look at Offline RL Agents

In the experiment, we re-implement following the baseline agents in JAX:

- TD3BC
- CQL
- COMBO
- IQL

## Evaluation Experiments

In this work, we take a closer look at the behaviors of current SOTA offline RL agents, especially for the learned representations, value functions and policies. Surprisingly, we find that the most performant offline RL agent sometimes has relatively low-quality representations and inaccurate value functions. In specific, a performant offline RL policy is usually able to select better sub-optimal actions, while avoiding bad ones.

### Representation Experiments

In this experiment, we run linear representation probing experiments and evaluate some recently proposed representation metrics.

```python
python exp_representation/run_exp.py
```

### Value Funtion Experiments

In this experiment, we evaluate the ability of the learned $Q$-function to rank different actions.

```python
python exp_value_function/value_ranking_exp.py
```

### Policy Experiments

In this experiment, we directly evaluate the learned policy of each agent.

```python
# policy ranking experiment
python exp_policy/policy_ranking_exp.py

# ood action experiment
python exp_policy/ood_action_pi_exp.py
```

## Relaxed In-smaple Q-Learning

We present a variant of IQL, which relaxes the in-sample constriant in the policy improvement step.

```shell
./relaxed_iql/run_exp.sh
```

## Uncertainty-based Sample Selection

We further investigate the use of a learned dynamics model for model-free offline RL agents.

```python
# run miql
python miql/main.py

# run mtd3bc
python mtd3bc/main.py
```

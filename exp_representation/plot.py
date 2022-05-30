import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['text.usetex'] = True

colors = [
    "#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#2ca02c", "#8c564b",
    "#e377c2", "#bcbd22", "#17becf"
]


# split a combined df to different seeds
def split_df_to_seeds(algo="iql", env_name="hopper-medium-v2"):
    df = pd.read_csv(f"res/{algo}_{env_name}.csv", index_col=0)
    for i in range(3):
        seed_df = df.query(f"seed == {i}").reset_index(drop=True)
        seed_df.to_csv(f"res/{algo}_s{i}_{env_name}.csv")


# split_df_to_seeds("iql", "halfcheetah-medium-v2")

ALGOS = ["td3bc", "cql", "combo", "iql"]


def compare_agents(env_name="halfcheetah-medium-v2"):
    _, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 4))
    cv_reward_cols = [f"cv_reward_loss{i}" for i in range(1, 6)]
    cv_next_obs_cols = [f"cv_next_obs_loss{i}" for i in range(1, 6)]
    cv_optimal_act_cols = [f"cv_optimal_action_loss{i}" for i in range(1, 6)]
    cv_optimal_q_cols = [f"cv_optimal_Q_loss{i}" for i in range(1, 6)]

    for algo in ALGOS:
        df = pd.read_csv(f"res/{algo}_s1_{env_name}.csv", index_col=0)
        axes[0][0].plot(df["step"], df["eval_reward"], label=algo)
        axes[0][0].set_title("reward")

        axes[0][1].plot(df["step"], df[cv_reward_cols].mean(1), label=algo)
        axes[0][1].set_title("reward loss")

        axes[0][2].plot(df["step"], df[cv_next_obs_cols].mean(1), label=algo)
        axes[0][2].set_title("next_obs loss")

        axes[0][3].plot(df["step"],
                        df[cv_optimal_act_cols].mean(1),
                        label=algo)
        axes[0][3].set_title("optimal_act loss")

        axes[1][0].plot(df["step"], df[cv_optimal_q_cols].mean(1), label=algo)
        axes[1][0].set_title("optimal_q loss")

        axes[1][1].plot(df["step"], df["sa_repr_norm_avg"], label=algo)
        axes[1][1].set_title("sa_repr_norm")

        axes[1][2].plot(df["step"], df["sa_srank"], label=algo)
        axes[1][2].set_title("sa_srank")

        axes[1][3].plot(df["step"], df["s_srank"], label=algo)
        axes[1][3].set_title("s_srank")

    plt.tight_layout()
    plt.savefig(f"compare_{env_name}.png")


def plot_probing_loss(env_name="halfcheetah-medium-v2", optimal_algo="combo"):
    _, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
    cv_reward_cols = [f"cv_reward_loss{i}" for i in range(1, 6)]
    cv_next_obs_cols = [f"cv_next_obs_loss{i}" for i in range(1, 6)]
    cv_inverse_act_cols = [f"cv_inverse_action_loss{i}" for i in range(1, 6)]
    cv_optimal_act_cols = [f"cv_optimal_action_loss{i}" for i in range(1, 6)]
    cv_optimal_q_cols = [f"cv_optimal_Q_loss{i}" for i in range(1, 6)]
    cv_optimal_v_cols = [f"cv_optimal_V_loss{i}" for i in range(1, 6)]

    plt_fields = [("reward probing loss", cv_reward_cols),
                  ("next state probing loss", cv_next_obs_cols),
                  ("inverse action probing loss", cv_inverse_act_cols),
                  ("optimal action probing loss", cv_optimal_act_cols),
                  (r"optimal $Q^*$ probing loss", cv_optimal_q_cols),
                  (r"optimal $V^*$ probing loss", cv_optimal_v_cols)]
    algos = ["td3bc", "cql", "combo", "iql"]
    for color_idx, algo in enumerate(algos):
        for plt_idx, (name, cols) in enumerate(plt_fields):
            ax = axes[plt_idx // 3][plt_idx % 3]
            if plt_idx % 3 == 0:
                ax.set_ylabel('MSE', fontsize=11)
            if plt_idx // 3 == 1:
                ax.set_xlabel('Time Steps', fontsize=11)
            if plt_idx in [4, 5]:
                ax.set_yscale('log')
            tmp_res = []
            for seed in range(5):
                df = pd.read_csv(f"res/{env_name}/{algo}/s{seed}.csv", index_col=0)
                tmp_res.append(df[cols].mean(axis=1).values.reshape(-1, 1))
            tmp_res = np.concatenate(tmp_res, axis=1)
            mu = tmp_res.mean(axis=1)
            std = tmp_res.std(axis=1)
            label = rf"{algo}($\star$)" if algo == optimal_algo else algo
            ax.plot(df.index * 100000, mu, color=colors[color_idx], label=label)
            ax.fill_between(df.index * 100000,
                            mu - std,
                            mu + std,
                            color=colors[color_idx],
                            alpha=0.08)
            ax.set_title(name)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"imgs/probing_exp/{env_name}.png", dpi=480)


def plot_repr_rank(env_name="halfcheetah-medium-v2", optimal_algo="combo"):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))
    plt_fields = [(r"$\phi(s, a)$ dot-product", "sa_dot_product"),
                  (r"$\phi(s, a)$ cosine similarity", "sa_cosine_similarity"),
                  (r"$\phi(s, a)$ effective rank", "sa_srank"),
                  (r"$\phi(s, a)$ effective dimension", "sa_eff_dim2")
                #   (r"$\psi(s)$ dot-product", "s_dot_product"),
                #   (r"$\psi(s)$ cosine similarity", "s_cosine_similarity"),
                #   (r"$\psi(s)$ effective rank", "s_srank")
                ]

    # plot online agent
    for plt_idx, (name, col) in enumerate(plt_fields):
        ax = axes[plt_idx]
        ax.set_xlabel('Time Steps', fontsize=11)
        if plt_idx == 0: ax.set_yscale("log")
        tmp_res = []
        for seed in range(5):
            if plt_idx == 3:
                df = pd.read_csv(f"res/{env_name}/td3/s{seed}_eff_dim_5W.csv", index_col=0)
            else:
                df = pd.read_csv(f"res/{env_name}/td3/s{seed}.csv", index_col=0)
            tmp_res.append(df[col].values.reshape(-1, 1))
        tmp_res = np.concatenate(tmp_res, axis=1)
        mu = tmp_res.mean(axis=1)
        std = tmp_res.std(axis=1)

        ax.plot(np.arange(11)*100000, np.ones(11)*mu, color=colors[6], label="online")
        ax.fill_between(np.arange(11)*100000, np.ones(11)*(mu-std), np.ones(11)*(mu+std), color=colors[6], alpha=0.08)

    # plot baseline agents
    for color_idx, algo in enumerate(ALGOS):
        for plt_idx, (name, col) in enumerate(plt_fields):
            ax = axes[plt_idx]
            if plt_idx == 0: ax.set_yscale("log")
            tmp_res = []
            for seed in range(5):
                if plt_idx == 3:
                    df = pd.read_csv(f"res/{env_name}/{algo}/s{seed}_eff_dim_5W.csv", index_col=0)
                else:
                    df = pd.read_csv(f"res/{env_name}/{algo}/s{seed}.csv", index_col=0)
                tmp_res.append(df[col].values.reshape(-1, 1))
            tmp_res = np.concatenate(tmp_res, axis=1)
            mu = tmp_res.mean(axis=1)
            std = tmp_res.std(axis=1)
            label = rf"{algo}($\star$)" if algo == optimal_algo else algo
            ax.plot(df.index * 100000, mu, color=colors[color_idx], label=label)
            ax.fill_between(df.index * 100000,
                            mu - std,
                            mu + std,
                            color=colors[color_idx],
                            alpha=0.08)
            ax.set_title(name)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"imgs/rank_exp/{env_name}.png", dpi=480)


if __name__ == "__main__":
    os.makedirs("imgs/probing_exp", exist_ok=True)
    os.makedirs("imgs/rank_exp", exist_ok=True)
    plot_probing_loss("halfcheetah-medium-v2", "combo")
    plot_repr_rank("halfcheetah-medium-v2", "combo")

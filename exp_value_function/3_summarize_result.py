import pandas as pd
from tqdm import tqdm


ALGOS = ['td3bc', 'cql', 'combo', 'iql']
ENVS = [
    "halfcheetah-medium-v2",
    "halfcheetah-medium-replay-v2",
    "halfcheetah-medium-expert-v2",
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
    "walker2d-medium-expert-v2",
]
COLS = ['rank_IC', 'top1_acc', 'top3_acc', 'top5_acc', 'top10_acc',
        'rank_IC2', 'top1_acc2', 'top3_acc2', 'top5_acc2', 'top10_acc2']

optimal_algo = {
    "halfcheetah-medium-v2": "combo",
    "halfcheetah-medium-replay-v2": "combo",
    "halfcheetah-medium-expert-v2": "iql",
    "hopper-medium-v2": "combo",
    "hopper-medium-replay-v2": "combo",
    "hopper-medium-expert-v2": "cql",
    "walker2d-medium-v2": "td3bc",
    "walker2d-medium-replay-v2": "cql",
    "walker2d-medium-expert-v2": "td3bc",
}


def save_mu_std_df():
    for env_name in tqdm(ENVS):
        # concat res from different `algo` and `seeds`
        env_res = []
        for seed in range(5):
            res_df = pd.read_csv(f"res/value_exp/rank_IC/{env_name}/s{seed}.csv")
            res_df["seed"] = seed
            env_res.append(res_df)
        env_res_df = pd.concat(env_res, axis=0).set_index(["algo", "seed"])

        # summary as mu/std
        mu_res, std_res = [], []
        for algo in ALGOS:
            for col in COLS:
                col_res = env_res_df.loc[algo].loc[:, col]
                mu_res.append((algo, col, col_res.mean()))
                std_res.append((algo, col, col_res.std()))

        # save mu/std res df
        mu_df = pd.DataFrame(mu_res, columns=["algo", "target", "value"])
        std_df = pd.DataFrame(std_res, columns=["algo", "target", "value"])
        mu_df.to_csv(f"res/value_exp/rank_IC/{env_name}/mu_res.csv")
        std_df.to_csv(f"res/value_exp/rank_IC/{env_name}/std_res.csv")


def color_brown(x): return "{\color{brown}" + x + "}"


def bold(x): return "\\textbf{" + x + "}"


def shorten(x):
    x = x.replace("medium", "med").replace("replay", "rep").replace("expert", "exp")
    return x


def print_ic_latex_table(target="rank_IC2", optimal_idx=-1):
    for env_name in ENVS:
        mu_df = pd.read_csv(f"res/value_exp/rank_IC/{env_name}/mu_res.csv", index_col=0).set_index(
            ["algo", "target"])
        std_df = pd.read_csv(f"res/value_exp/rank_IC/{env_name}/std_res.csv", index_col=0).set_index(
            ["algo", "target"])

        ic_res = sorted([(algo, mu_df.loc[(algo, target), "value"]) for algo in ALGOS],
                        key=lambda x: x[1])

        log_res = f"{shorten(env_name)}"
        for algo in ALGOS:
            mu = mu_df.loc[(algo, target), "value"]
            std = std_df.loc[(algo, target), "value"]

            log_res += " & "
            if algo == optimal_algo[env_name]:
                if algo == ic_res[optimal_idx][0]:
                    log_res += bold(color_brown(f"{mu:.2f} ({std:.2f})"))
                else:
                    log_res += color_brown(f"{mu:.2f} ({std:.2f})")
            else:
                if algo == ic_res[optimal_idx][0]:
                    log_res += bold(f"{mu:.2f} ({std:.2f})")
                else:
                    log_res += f"{mu:.2f} ({std:.2f})"
        log_res += " \\\\\n"
        print(log_res)


def print_acc_latex_table():
    for env_name in ENVS:
        mu_df = pd.read_csv(f"res/value_exp/rank_IC/{env_name}/mu_res.csv", index_col=0).set_index(
            ["algo", "target"]) * 100
        std_df = pd.read_csv(f"res/value_exp/rank_IC/{env_name}/std_res.csv", index_col=0).set_index(
            ["algo", "target"])

        acc_res = sorted([(algo, mu_df.loc[(algo, "top5_acc2"), "value"]) for algo in ALGOS],
                         key=lambda x: x[1])

        log_res = f"{shorten(env_name)}"
        for algo in ALGOS:
            mu1 = mu_df.loc[(algo, "top1_acc2"), "value"]
            std1 = std_df.loc[(algo, "top1_acc2"), "value"]
            mu2 = mu_df.loc[(algo, "top5_acc2"), "value"]
            std2 = std_df.loc[(algo, "top5_acc2"), "value"]

            log_res += " & "
            mu_std1 = f"{mu1:.2f}"
            mu_std2 = f"{mu2:.2f}"
            if algo == optimal_algo[env_name]:
                if algo == acc_res[-1][0]:
                    log_res += bold(color_brown(mu_std1)) + " & " + bold(color_brown(mu_std2))
                else:
                    log_res += color_brown(mu_std1) + " & " + color_brown(mu_std2)
            else:
                if algo == acc_res[-1][0]:
                    log_res += bold(mu_std1) + " & " + bold(mu_std2)
                else:
                    log_res += mu_std1 + " & " + mu_std2
        log_res += " \\\\\n"
        print(log_res)


if __name__ == "__main__":
    save_mu_std_df()
    print_ic_latex_table("rank_IC2", -1)
    print_acc_latex_table()

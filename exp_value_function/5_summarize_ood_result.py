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
COLS = ['ood_Qs_25q', 'ood_Qs_50q', 'ood_Qs_75q',
        'in_sample_Qs_25q', 'in_sample_Qs_50q', 'in_sample_Qs_75q',
        'd4rl_Qs_25q', 'd4rl_Qs_50q', 'd4rl_Qs_75q']
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


def color_brown(x):
    return " {\color{brown}" + x + "}"


def bold(x):
    return "\\textbf{" + x + "}"


def shorten(x):
    x = x.replace("medium", "med").replace("replay", "rep").replace("expert", "exp")
    return x


def print_latex_table0():
    for env_name in ENVS:
        mu_df = pd.read_csv(f"res/ood_value/{env_name}/mu_res.csv", index_col=0).set_index(
            ["algo"])
        std_df = pd.read_csv(f"res/ood_value/{env_name}/std_res.csv", index_col=0).set_index(
            ["algo"])

        log_res = f"{shorten(env_name)}"
        for algo in ALGOS:
            mu1 = mu_df.loc[algo, "ratio_25q"]
            std1 = std_df.loc[algo, "ratio_25q"]
            mu2 = mu_df.loc[algo, "ratio_50q"]
            std2 = std_df.loc[algo, "ratio_50q"]

            log_res += " & "
            mu_std1 = f"{mu1:.2f} ({std1:.2f})"
            mu_std2 = f"{mu2:.2f} ({std2:.2f})"
            if algo == optimal_algo[env_name]:
                log_res += color_brown(mu_std1) + " & " + color_brown(mu_std2)
            else:
                log_res += mu_std1 + " & " + mu_std2
        log_res += " \\\\\n"
        print(log_res)


def print_latex_table():
    for env_name in ENVS:
        mu_df = pd.read_csv(f"res/ood_value/{env_name}/mu_res.csv", index_col=0).set_index(
            ["algo"])
        std_df = pd.read_csv(f"res/ood_value/{env_name}/std_res.csv", index_col=0).set_index(
            ["algo"])

        log_res = f"{shorten(env_name)}"
        for algo in ALGOS:
            mu = mu_df.loc[algo, "ood_Qs_50q"]
            std = std_df.loc[algo, "ood_Qs_50q"]

            log_res += " & "
            if algo == optimal_algo[env_name]:
                log_res += color_brown(f"{mu:.2f} ({std:.2f})")
            else:
                log_res += f"{mu:.2f} ({std:.2f})"
        log_res += " \\\\\n"
        print(log_res)




if __name__ == "__main__":
    print_latex_table()

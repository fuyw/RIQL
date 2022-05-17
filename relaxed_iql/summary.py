import os
import numpy as np
import pandas as pd
ENVS = [f"{task}-{level}-v2" for task in ["halfcheetah", "hopper", "walker2d"] for level in ["medium", "medium-replay", "medium-expert"]]


def read_csvs(fdir, file_names):
    idx = range(955000, 1005000, 5000)
    rew_res = []
    for file_name in file_names:
        df = pd.read_csv(f"{fdir}/{file_name}", index_col=0).set_index("step")
        assert len(df) == 106 
        reward = df.loc[idx, 'reward'].mean()
        rew_res.append(reward)
    rew_res = np.array(rew_res)
    mu, std = rew_res.mean(), rew_res.std()
    return mu, std


def get_niql_res(name="agent2"):
    res = []
    for env in ENVS:
        fdir = f"logs/{env}"
        file_names = [i for i in os.listdir(f"logs/{env}") if f"{name}_" in i and ".csv" in i]
        mu, std = read_csvs(fdir, file_names)
        # res.append((env, mu, std))
        res.append((env, mu))
    # res_df = pd.DataFrame(res, columns=["env_name", "mu", "std"]).set_index(["env_name"])
    res_df = pd.DataFrame(res, columns=["env_name", name]).set_index(["env_name"])
    return res_df


res_df = get_niql_res("agent2")
res_df["iql"] = [47.43, 44.04, 88.92, 64.64, 91.94, 91.57, 80.28, 73.53, 107.68]
res_df["td3bc"] = [48.89, 45.20, 89.29, 60.19, 64.96, 96.53, 84.37, 77.24, 110.16]
print(res_df.round(2))

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


COLS = [
    # basic info 
    'eval_reward',

    # effective dimension
    'sa_eff_dim1', 'sa_eff_dim2', 's_eff_dim1', 's_eff_dim2',
    
    # effective rank
    'sa_srank', 's_srank',

    # embedding norm
    'sa_embeddings_norm', #'sa_embeddings_norm_std', 'sa_embeddings_norm_min',  'sa_embeddings_norm_max', 
    's_embeddings_norm', #'s_embeddings_norm_std', 's_embeddings_norm_min', 's_embeddings_norm_max',

    # dot-product     
    'sa_dot_product', 'sa_cosine_similarity', 's_dot_product', 's_cosine_similarity',
    
    # fixed-q 
    # 'fixed_q', 'fixed_q_max', 'fixed_q_min', 'fixed_q_std',
    
    # kernel/output norm
    'kernel_norm', 'output_norm',
]
PROBING_DICT = {
    # probe reward   
    'reward probing loss': ['cv_reward_loss1', 'cv_reward_loss2', 'cv_reward_loss3', 'cv_reward_loss4', 'cv_reward_loss5'],
        
    # probe next_obs
    'next state probing loss': ['cv_next_obs_loss1', 'cv_next_obs_loss2', 'cv_next_obs_loss3', 'cv_next_obs_loss4', 'cv_next_obs_loss5'],

    # probe inverse_action
    'inverse action probing loss': ['cv_inverse_action_loss1', 'cv_inverse_action_loss2', 'cv_inverse_action_loss3', 'cv_inverse_action_loss4', 'cv_inverse_action_loss5'],
        
    # probe optimal_action
    'optimal action probing loss': ['cv_optimal_action_loss1', 'cv_optimal_action_loss2', 'cv_optimal_action_loss3', 'cv_optimal_action_loss4', 'cv_optimal_action_loss5'],

    # probe optimal_Q
    'optimal Q probing loss': ['cv_optimal_Q_loss1', 'cv_optimal_Q_loss2', 'cv_optimal_Q_loss3', 'cv_optimal_Q_loss4', 'cv_optimal_Q_loss5'],
       
    # probe optimal_V
    'optimal V probing loss': ['cv_optimal_V_loss1', 'cv_optimal_V_loss2', 'cv_optimal_V_loss3', 'cv_optimal_V_loss4', 'cv_optimal_V_loss5']
}


def save_mu_std_df():
    for env_name in tqdm(ENVS):
        # concat res from different `algo` and `seeds`
        env_res = []
        for seed in range(5):
            for algo in ALGOS:
                eff_dim_df = pd.read_csv(f'res/{env_name}/{algo}/s{seed}_eff_dim_5W.csv', index_col=0).iloc[-1, :]
                probe_exp_df = pd.read_csv(f'res/{env_name}/{algo}/s{seed}.csv', index_col=0).iloc[-1, :]
                eff_dim_df['seed'] = seed
                eff_dim_df['algo'] = algo
                concat_df = pd.concat([eff_dim_df, probe_exp_df], axis=0)
                env_res.append(concat_df)
        env_res_df = pd.concat(env_res, axis=1).T
        env_res_df['seed'] = env_res_df['seed'].astype('int')
        env_res_df = env_res_df.set_index(['algo', 'seed'])

        # summary as mu/std
        mu_res, std_res = [], []
        for algo in ALGOS:
            for col in COLS:
                col_res = env_res_df.loc[algo].loc[:, col]
                mu_res.append((algo, col, col_res.mean()))
                std_res.append((algo, col, col_res.std()))

            for target in PROBING_DICT:
                target_cv_loss = env_res_df.loc[algo].loc[:, PROBING_DICT[target]].mean(axis=1)
                mu_res.append((algo, target, target_cv_loss.mean()))
                std_res.append((algo, target, target_cv_loss.std()))

        # save mu/std res df
        mu_df = pd.DataFrame(mu_res, columns=["algo", "target", "value"])
        std_df = pd.DataFrame(std_res, columns=["algo", "target", "value"])
        mu_df.to_csv(f"res/{env_name}/mu_res.csv")
        std_df.to_csv(f"res/{env_name}/std_res.csv")


def color_brown(x):
    return " {\color{brown}" + x + "}"


def bold(x):
    return "\\textbf{" + x + "}"


def shorten(x):
    x = x.replace("medium", "med").replace("replay", "rep").replace("expert", "exp")
    return x


def print_latex_table(target="reward probing loss", optimal_idx=0):
    match_num = 0
    for env_name in ENVS:
        mu_df = pd.read_csv(f"res/{env_name}/mu_res.csv", index_col=0).set_index(["algo", "target"])
        std_df = pd.read_csv(f"res/{env_name}/std_res.csv", index_col=0).set_index(["algo", "target"])

        env_rewards = sorted([(algo, mu_df.loc[(algo, "eval_reward"), "value"]) for algo in ALGOS], key=lambda x: x[1])

        env_loss = sorted([(algo, mu_df.loc[(algo, target), "value"]) for algo in ALGOS], key=lambda x: x[1])

        log_res = f"{shorten(env_name)}"
        for algo in ALGOS:
            mu = mu_df.loc[(algo, target), "value"]
            std = std_df.loc[(algo, target), "value"]
            if 'optimal Q' in target or 'optimal V' in target or 'dot_product' in target:
                mu /= 1000
                std /= 1000

            log_res += " & "
            if algo == env_rewards[-1][0]:
                if algo == env_loss[optimal_idx][0]:
                    log_res += bold(color_brown(f"{mu:.2f} ({std:.2f})"))
                else:
                    log_res += color_brown(f"{mu:.2f} ({std:.2f})")
            else:
                if algo == env_loss[optimal_idx][0]:
                    log_res += bold(f"{mu:.2f} ({std:.2f})")
                else:
                    log_res += f"{mu:.2f} ({std:.2f})"

        log_res += " \\\\\n"
        # print(log_res)

        if env_rewards[-1][0] == env_loss[optimal_idx][0]:
            match_num += 1
    print(f"Match num: {match_num}/{len(ENVS)}")
    return match_num


if __name__ == "__main__":
    match_res = []
    for target in ["reward probing loss", "next state probing loss", "inverse action probing loss",
                   "optimal action probing loss", "optimal Q probing loss", "optimal V probing loss"]:
        match_res.append((target, print_latex_table(target, 0)))
    print(match_res)
    # print_latex_table('sa_eff_dim2', 0)
    # print_latex_table('sa_srank', -1)
    # print_latex_table('sa_dot_product', 0)
    # print_latex_table('sa_cosine_similarity', 0)


    # print_latex_table('s_eff_dim2', 0)
    # print_latex_table('s_srank', -1)
    # print_latex_table('s_dot_product', 0)
    # print_latex_table('s_cosine_similarity', 0)
# [('reward probing loss', 0),
#  ('next state probing loss', 0),
#  ('inverse action probing loss', 3),
#  ('optimal action probing loss', 5),
#  ('optimal Q probing loss', 4),
#  ('optimal V probing loss', 4)]

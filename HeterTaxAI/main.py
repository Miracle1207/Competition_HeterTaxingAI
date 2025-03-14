import numpy as np
from env.env_core import economic_society
from agents.rule_based import rule_agent
from agents.ddpg_agent import ddpg_agent
from agents.maddpg_agent import maddpg_agent
from agents.real_data.real_data import real_agent
from agents.mfrl import mfrl_agent
from agents.bi_mfrl import bi_mfrl_agent
from agents.bi_ddpg_agent import bi_ddpg_agent
from agents.aie_agent import aie_agent
from utils.seeds import set_seeds
import os
import argparse
from omegaconf import OmegaConf
from runner import Runner
from agents.submission import your_agent
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='default')
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--heterogeneous_house_agent", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--br", action='store_true')
    parser.add_argument("--bc", action='store_true')
    parser.add_argument("--house_alg", type=str, default='mfac', help="rule_based, ippo, mfac, real")
    parser.add_argument("--gov_alg", type=str, default='ac', help="ac, rule_based, independent")
    parser.add_argument("--task", type=str, default='gdp', help="gini, social_welfare, gdp_gini")
    parser.add_argument('--device-num', type=int, default=1, help='the number of cuda service num')
    parser.add_argument('--n_households', type=int, default=100, help='the number of total households')
    parser.add_argument('--seed', type=int, default=1, help='the random seed')
    parser.add_argument('--hidden_size', type=int, default=128, help='[64, 128, 256]')
    parser.add_argument('--q_lr', type=float, default=3e-4, help='[3e-3, 3e-4, 3e-5]')
    parser.add_argument('--p_lr', type=float, default=3e-4, help='[3e-3, 3e-4, 3e-5]')
    parser.add_argument('--batch_size', type=int, default=64, help='[32, 64, 128, 256]')
    parser.add_argument('--update_cycles', type=int, default=100, help='[10，100，1000]')
    parser.add_argument('--update_freq', type=int, default=10, help='[10，20，30]')
    parser.add_argument('--initial_train', type=int, default=10, help='[10，100，200]')
    args = parser.parse_args()
    return args


def select_agent(alg, agent_name):
    if agent_name == "households":
        if alg == "real":
            house_agent = real_agent(env, yaml_cfg.Trainer)
        elif alg == "mfrl":
            house_agent = mfrl_agent(env, yaml_cfg.Trainer)
        elif alg == "bi_mfrl":
            house_agent = bi_mfrl_agent(env, yaml_cfg.Trainer)
        elif alg == "ppo":
            house_agent = ppo_agent(env, yaml_cfg.Trainer, agent_name="household")
        elif alg == "ddpg":
            house_agent = ddpg_agent(env, yaml_cfg.Trainer, agent_name="household")
        elif alg == "maddpg":
            house_agent = maddpg_agent(env, yaml_cfg.Trainer, agent_name="household")
        elif alg == "rule_based":
            house_agent = rule_agent(env, yaml_cfg.Trainer)
        elif alg == "aie":
            house_agent = aie_agent(env, yaml_cfg.Trainer)
        elif alg == "your_policy":
            house_agent = your_agent(env, yaml_cfg.Trainer)
        else:
            print("Wrong Choice!")
        return house_agent
    else:
        if alg == "rule_based" or alg == "us_federal" or alg == "saez":
            gov_agent = rule_agent(env, yaml_cfg.Trainer)
        # elif alg == "ppo":
        #     gov_agent = ppo_agent(env, yaml_cfg.Trainer, agent_name="government")
        elif alg == "ddpg":
            gov_agent = ddpg_agent(env, yaml_cfg.Trainer, agent_name="government")
        elif alg == "maddpg":
            gov_agent = maddpg_agent(env, yaml_cfg.Trainer, agent_name="government")
        elif alg == "bi_ddpg":
            gov_agent = bi_ddpg_agent(env, yaml_cfg.Trainer, agent_name="government")
        elif alg == "aie":
            gov_agent = aie_agent(env, yaml_cfg.Trainer, agent_name="government")
        elif alg == "your_policy":
            gov_agent = your_agent(env, yaml_cfg.Trainer)
        else:
            print("Wrong Choice!")
        return gov_agent

def choose_model_path(house_alg, gov_alg, households_num):
    if house_alg == "bi_mfrl" and gov_alg == "bi_ddpg":
        # bi_mfrl_bi_ddpg
        if households_num == 100:
            house_model_path="agents/models/bi_mfrl_bi_ddpg/100/gdp/run129/bimf_house_actor.pt"
            government_model_path="agents/models/bi_mfrl_bi_ddpg/100/gdp/run39/bi_ddpg_net.pt"
        elif households_num == 1000:
            house_model_path="agents/models/bi_mfrl_bi_ddpg/1000/gdp/run15/bimf_house_actor.pt" # seed 10
            government_model_path="agents/models/bi_mfrl_bi_ddpg/1000/gdp/run15/bi_ddpg_net.pt"
    if house_alg == "aie" and gov_alg == "aie":
        if households_num == 100:
            house_model_path="agents/models/aie_aie/100/gdp/run64/household_aie_net.pt"
            government_model_path="agents/models/aie_aie/100/gdp/run64/government_aie_net.pt"
        elif households_num == 1000:
            house_model_path="agents/models/bi_mfrl_bi_ddpg/1000/gdp/run15/bimf_house_actor.pt"
            government_model_path="agents/models/bi_mfrl_bi_ddpg/1000/gdp/run15/bi_ddpg_net.pt"
    if house_alg == "maddpg" and gov_alg == "maddpg":
        if households_num == 100:
            # maddpg
            house_model_path="agents/models/maddpg_maddpg/100/gdp/run28/household_ddpg_net.pt"
            government_model_path="agents/models/maddpg_maddpg/100/gdp/run28/government_ddpg_net.pt"
        elif households_num == 1000:
            # maddpg
            house_model_path="agents/models/maddpg_maddpg/1000/gdp/run7/household_ddpg_net.pt"  # run10 - seed 11
            government_model_path="agents/models/maddpg_maddpg/1000/gdp/run7/government_ddpg_net.pt"  # run7 -seed 2
    # mfrl+ddpg
    if house_alg == "mfrl" and gov_alg == "ddpg":
        if households_num == 100:
            house_model_path="agents/models/mfrl_ddpg/100/gdp/run15/house_actor.pt"
            government_model_path="agents/models/mfrl_ddpg/100/gdp/run15/ddpg_net.pt"
        elif households_num == 1000:
            house_model_path="agents/models/mfrl_ddpg/1000/gdp/run4/house_actor.pt"  # run2 -seed2
            government_model_path="agents/models/mfrl_ddpg/1000/gdp/run4/government_ddpg_net.pt"  # run4 - seed1
    # iddpg
    if house_alg == "ddpg" and gov_alg == "ddpg":
        if households_num == 100:
            house_model_path="agents/models/ddpg_ddpg/100/gdp/run15/household_ddpg_net.pt"
            government_model_path="agents/models/ddpg_ddpg/100/gdp/run15/government_ddpg_net.pt"
        elif households_num == 1000:
            house_model_path="agents/models/ddpg_ddpg/1000/gdp/run7/household_ddpg_net.pt"
            government_model_path="agents/models/ddpg_ddpg/1000/gdp/run7/government_ddpg_net.pt"
    if (house_alg == "real" or house_alg == "rule_based"):
        house_model_path= None
    elif house_alg == "bi_mfrl":
        if households_num == 100:
            house_model_path="agents/models/bi_mfrl_bi_ddpg/100/gdp/run129/bimf_house_actor.pt"
      
    if gov_alg == "bi_ddpg":
        if households_num == 100:
            government_model_path = "agents/models/bi_mfrl_bi_ddpg/100/gdp/run39/bi_ddpg_net.pt"
    elif gov_alg == "ddpg":
        if households_num == 100:
            government_model_path="agents/models/mfrl_ddpg/100/gdp/run15/ddpg_net.pt"
    elif gov_alg == "rule_based" or gov_alg == "saez" or gov_alg == "us_federal":
        government_model_path = None


    return house_model_path, government_model_path



if __name__ == '__main__':
    # Set single-thread execution
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    # Parse arguments
    args = parse_args()
    path = args.config
    yaml_cfg = OmegaConf.load(f'./cfg/{path}.yaml')
    # args = Namespace(config='default', wandb=False, heterogeneous_house_agent=True, test=True, br=False, bc=False,
    #                  house_alg='bi_mfrl', gov_alg='ddpg', task='gdp', device_num=1, n_households=100, seed=1,
    #                  hidden_size=128, q_lr=0.0003, p_lr=0.0003, batch_size=128, update_cycles=100, update_freq=10,
    #                  initial_train=10)

    # Update Trainer configurations
    # 更新训练器配置
    yaml_cfg.Trainer.update({
        "n_households": args.n_households,
        "seed": args.seed,
        "bc": args.bc,
        "wandb": args.wandb,
        "find_best_response": args.br,
        "heterogeneous_house_agent": args.heterogeneous_house_agent,
        "hidden_size": args.hidden_size,
        "q_lr": args.q_lr,
        "p_lr": args.p_lr,
        "batch_size": args.batch_size,
        "house_alg": args.house_alg,
        "gov_alg": args.gov_alg
    })

    # Check and update Environment configurations
    # 检查并更新环境配置
    if len(yaml_cfg.Environment.Entities) > 1:
        yaml_cfg.Environment.Entities[1]["entity_args"]["n"] = args.n_households
    else:
        print("Warning: Entities list does not have enough elements.")

    if "env_args" in yaml_cfg.Environment.env_core:
        yaml_cfg.Environment.env_core["env_args"]["gov_task"] = args.task
    else:
        print("Warning: 'env_args' not found in 'env_core'.")

    if args.gov_alg in ["saez", "us_federal"]:
        if "env_args" in yaml_cfg.Environment["env_core"]:
            yaml_cfg.Environment["env_core"]["env_args"]["tax_module"] = args.gov_alg
        else:
            print("Warning: 'env_args' not found in 'env_core'.")

    # Set seed and CUDA device
    set_seeds(args.seed, cuda=yaml_cfg.Trainer["cuda"])
    if args.device_num is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_num)

    # Initialize environment and agents
    env = economic_society(yaml_cfg.Environment)
    house_agent = select_agent(args.house_alg, agent_name="households")
    gov_agent = select_agent(args.gov_alg, agent_name="government")

    print("n_households: ", yaml_cfg.Trainer["n_households"])

    # Test or run training
    # test = True
    if args.test:
        heter_house_agents = None
        if args.heterogeneous_house_agent:
            # Predefined heterogeneous households' policies for competition
            SMFG_households = select_agent('bi_mfrl', agent_name="households")
            bc_households = select_agent('real', agent_name="households")
            random_households = select_agent('rule_based', agent_name="households")
            heter_house_agents = [SMFG_households, bc_households, random_households]


        # house_agent = < agents.bi_mfrl.bi_mfrl_agent object at 0x00000272D98642E0 >
        # gov_agent = < agents.ddpg_agent.ddpg_agent object at 0x00000272DCE81610 >

        # yaml_cfg.Trainer = {'n_households': 100, 'log_std_min': -20, 'log_std_max': 2, 'hidden_size': 128,
        #                     'cuda': False, 'q_lr': 0.0003, 'p_lr': 0.0003, 'buffer_size': 1000000.0,
        #                     'env_name': 'wealth_distribution', 'init_exploration_policy': 'gaussian', 'n_epochs': 500,
        #                     'epoch_length': 300, 'update_cycles': 100, 'target_update_interval': 1,
        #                     'display_interval': 1, 'batch_size': 128, 'gamma': 0.975, 'tau': 0.95, 'eval_episodes': 5,
        #                     'init_exploration_steps': 1000, 'ppo_tau': 0.95, 'ppo_gamma': 0.99, 'eps': 1e-05,
        #                     'update_epoch': 20, 'clip': 0.1, 'vloss_coef': 0.5, 'ent_coef': 0.01, 'max_grad_norm': 0.5,
        #                     'update_freq': 2, 'initial_train': 100, 'noise_rate': 0.01, 'epsilon': 0.1,
        #                     'save_interval': 100, 'house_alg': 'bi_mfrl', 'gov_alg': 'ddpg', 'update_each_epoch': 100,
        #                     'seed': 1, 'wandb': False, 'best_response': False, 'entropy_coef': 0.025, 'bc': False,
        #                     'heterogeneous_house_agent': True, 'heter_house_alg': 'real', 'heter_house_rate': 0,
        #                     'economic_shock': 'None', 'find_best_response': False}
        runner = Runner(env, yaml_cfg.Trainer, house_agent=house_agent, government_agent=gov_agent,
                        heter_house=heter_house_agents)

        house_model_path, government_model_path = choose_model_path(
            house_alg=args.house_alg, gov_alg=args.gov_alg, households_num=args.n_households
        )
        # house_model_path = agents / models / bi_mfrl_bi_ddpg / 100 / gdp / run129 / bimf_house_actor.pt
        # government_model_path = agents / models / mfrl_ddpg / 100 / gdp / run15 / ddpg_net.pt

        # runner.test(house_model_path, government_model_path)
        if args.heterogeneous_house_agent:  # heterogeneous_house_agent=True
            # todo: for compeition
            # track = "households"  # 家庭赛道
            track = "government"
            if track == "households":
                multiple_heter_agents_numbers = [
                    [100, 100, 100, 100],
                    [100, 100, 100, 100],
                    [70, 80, 90, 100],
                    [40, 60, 80, 100],
                    [25, 50, 75, 100]]

                # house_model_path = agents / models / bi_mfrl_bi_ddpg / 100 / gdp / run129 / bimf_house_actor.pt
                # government_model_path = agents / models / mfrl_ddpg / 100 / gdp / run15 / ddpg_net.pt
                scores_dict = runner.test_heter_agents(house_model_path, government_model_path,
                                                       multiple_heter_agents_numbers)
                print("households track evaluation indicators -- Your_agent_average_utility: ", scores_dict["Your_agent_average_utility"])
            elif track == "government":
                # with your households agent set as bi_mfrl
                multiple_heter_agents_numbers = [[0, 34, 67, 100],
                                                 [0, 80, 90, 100],
                                                 [0, 10, 90, 100],
                                                 [0, 10, 20, 100]]
                scores_dict = runner.test_heter_agents(house_model_path, government_model_path, multiple_heter_agents_numbers)
                print("government track evaluation indicators -- Government_return: ", scores_dict["Government_return"])
            print(scores_dict)
        else:
            runner.test(house_model_path, government_model_path)
    else:
        # Ensure yaml_cfg.Trainer is a mutable dictionary
        yaml_cfg.Trainer = OmegaConf.to_container(yaml_cfg.Trainer, resolve=True)
        runner = Runner(env, yaml_cfg.Trainer, house_agent=house_agent, government_agent=gov_agent)
        runner.run()


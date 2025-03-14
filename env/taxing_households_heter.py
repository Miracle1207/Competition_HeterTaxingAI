import os
import sys
import random
from pathlib import Path
import numpy as np
from utils.box import Box
from omegaconf import OmegaConf
from env.simulators.game import Game
from HeterTaxAI.env.env_core import economic_society
CURRENT_PATH = str(Path(__file__).resolve().parent.parent)
taxing_path = os.path.join(CURRENT_PATH)
sys.path.append(taxing_path)

__all__ = ['Taxing_Household_Heter']


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)
    
    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)
    
    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=Flase
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        return x

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.agent_name = 'Random'
    def __call__(self, observation):
        return [self.action_space[0].sample()]


class Taxing_Household_Heter(Game):
    def __init__(self, conf, seed=None):
        super(Taxing_Household_Heter, self).__init__(conf['n_player'], conf['is_obs_continuous'], conf['is_act_continuous'],
                                               conf['game_name'], conf['agent_nums'], conf['obs_type'])
        self.seed = seed
        self.set_seed()
        self.heter_house_number = [40, 60, 80, 100]
        yaml_path = os.path.join(CURRENT_PATH, 'HeterTaxAI/cfg/n100.yaml')
        yaml_cfg = OmegaConf.load(yaml_path)
        self.env_core = economic_society(yaml_cfg.Environment)
        self.max_step = int(conf['max_step'])
        self.controllable_agent_id = 'Household'
        self.agent_id = list(self.env_core.action_spaces.keys())
        self.joint_action_space = {self.controllable_agent_id :self.set_action_space()[self.controllable_agent_id]}
        self.action_dim = self.joint_action_space
        self.each_gov_spaces = Box(np.array([-1., -1., -1, -1, -1]), np.array([1.,1., 1., 1., 1.]))
        self.each_household_spaces = Box(-1.0, 1.0, (100, 2), np.float32)
        self.n_households = self.env_core.households.n_households
        self.n_gov = 1
        self.sub_controller_gov = [RandomAgent([self.each_gov_spaces])]
        self.sub_controller_household = [RandomAgent([self.each_household_spaces]) for i in range(3)]

        self.reset()
        self.init_info = {'Controllable': self.controllable_agent_id,
                          'Opponent': [i.agent_name for i in self.sub_controller_gov]}

    @staticmethod
    def create_seed():
        seed = random.randrange(1000)
        return seed

    def set_seed(self, seed=None):
        if not seed:        # use previous seed when no new seed input
            seed = self.seed
        else:               # update env global seed
            self.seed = seed
        random.seed(seed)
        np.random.seed(seed)


    def set_action_space(self):
        a_s = {}
        for aid, space in self.env_core.action_spaces.items():
            a_s[aid] = [space]
        return a_s

    def get_single_action_space(self, player_id):
        player_name = self.controllable_agent_id
        return self.joint_action_space[player_name]

    def reset(self):
        global_obs, private_obs = self.env_core.reset()
        self.global_state_norm = Normalization(shape=6)
        self.private_state_norm = Normalization(shape=2)
        global_obs = self.global_state_norm(global_obs)
        private_obs = self.private_state_norm(private_obs)
        self.step_cnt = 0
        self.done = False
        self.init_info = None
        self.won = {}
        self.n_return = [0]*self.n_player
        self.total_r = []

        self.current_state = (global_obs, private_obs)
        self.all_observes = self.get_all_observes()
        return self.all_observes

    def step(self, joint_action):

        self.is_valid_action(joint_action)
        joint_action_decode = self.decode(joint_action, self.heter_house_number)
        info_before = {"actions": joint_action_decode}

        global_obs, private_obs, gov_r, house_r, done = self.env_core.step(joint_action_decode)
        info_after = self.step_after_info()
        self.current_state = (global_obs, private_obs)
        self.all_observes = self.get_all_observes()

        reward = np.mean(house_r[:self.heter_house_number[0]], axis=0)
        self.total_r.append(reward)

        self.step_cnt += 1
        self.done = done
        if self.done:
            self.set_n_return()
            print('Final n_return = ', self.n_return)

        return self.all_observes, reward, self.done, info_before, info_after


    def is_valid_action(self, joint_action):
        if len(joint_action) != self.n_player:          # check number of player
            raise Exception("Input joint action dimension should be {}, not {}".format(
                self.n_player, len(joint_action)))

    def decode(self, joint_action, heter_house_number):
        joint_action_decode = {}

        for gov_idx, gov_obs in enumerate(self.gov_obs):

            _gov_action = self.sub_controller_gov[gov_idx](gov_obs)[0]

        joint_action_decode['government'] = _gov_action

        household_action = []
        household_action.append(joint_action[0][0][:heter_house_number[0]])
        self.all_observes[0]['gov_action'] = joint_action_decode['government']

        for household_idx, each_heter_agent in enumerate(self.sub_controller_household):
            _action = each_heter_agent(self.all_observes[0])[0]
            household_action.append(_action[heter_house_number[household_idx]:heter_house_number[household_idx + 1]])

        household_action = np.vstack(household_action)
        joint_action_decode[self.controllable_agent_id] = household_action

        return joint_action_decode


    def get_all_observes(self):
        all_observes = []
        global_obs, private_obs = self.current_state
        global_obs = self.global_state_norm(global_obs)
        private_obs = self.private_state_norm(private_obs)

        # global obs is for government, private_obs+global_obs are for each household
        gov_obs = []

        for idx, aid in enumerate(self.agent_id):
            if aid == 'government':
                _gov_obs = {'obs': {"agent_id": aid, "raw_obs": np.copy(global_obs)},
                           "controlled_player_index": idx}
                gov_obs.append(_gov_obs)

            elif aid == 'Household':
                _obs = {"obs": {"agent_id": aid, "raw_obs": [np.copy(global_obs), np.copy(private_obs)]},
                        "controlled_player_index": idx}
                all_observes.append(_obs)
            else:
                raise NotImplementedError

        self.gov_obs = gov_obs

        return all_observes

    def step_after_info(self):
        current_step = self.env_core.step_cnt
        social_welfare = self.env_core.households_reward.mean()
        wealth_gini = self.env_core.wealth_gini
        income_gini = self.env_core.income_gini
        gdp = self.env_core.GDP

        tau = self.env_core.government.tau
        xi = self.env_core.government.xi
        tau_a = self.env_core.government.tau_a
        xi_a = self.env_core.government.xi_a
        Gt_prob = self.env_core.Gt_prob
        return {'step': current_step, 'social_welfare': social_welfare, 'wealth_gini': wealth_gini,
                'income_gini': income_gini, 'gdp': gdp, 'tau': tau, 'xi': xi, 'tau_a': tau_a, 'xi_a': xi_a,
                "Gt_prob": Gt_prob}

    def is_terminal(self):
        return self.done

    def check_win(self):
        return '-1'

    def set_n_return(self):
        self.n_return = np.sum(self.total_r)
        self.total_r.clear()

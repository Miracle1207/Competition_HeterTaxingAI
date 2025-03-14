import copy
import numpy as np
import torch
import os, sys
import wandb
import json
sys.path.append(os.path.abspath('../..'))
from agents.log_path import make_logpath
from utils.experience_replay import replay_buffer
from datetime import datetime
import time

torch.autograd.set_detect_anomaly(True)

class Runner:
    def __init__(self, envs, args, house_agent, government_agent, heter_house=None):
        self.envs = envs   # 环境对象，用于与智能体进行交互
        self.args = args  # 参数配置，包含训练和环境的相关参数
        self.eval_env = copy.copy(envs)  # 评估环境，用于评估智能体的性能
        self.house_agent = house_agent  # 家庭智能体，负责家庭的决策
        self.government_agent = government_agent  # 政府智能体，负责政府的决策
        if self.args.heterogeneous_house_agent == True:  # 如果使用异构家庭智能体
            self.heter_house = heter_house  # 异构家庭智能体列表
        # define the replay buffer
        self.buffer = replay_buffer(self.args.buffer_size)  # 定义经验回放缓冲区，用于存储经验以供off-policy算法使用
        # state normalization
        # 政府观察空间归一化
        self.global_state_norm = Normalization(shape=self.envs.government.observation_space.shape[0])   # 状态归一化处理，确保输入到智能体的观测值具有合适的数值范围
        # 家庭观察空间归一化
        self.private_state_norm = Normalization(shape=self.envs.households.observation_space.shape[0])  # 状态归一化处理，确保输入到智能体的观测值具有合适的数值范围
        # 日志和模型路径初始化
        self.model_path, _ = make_logpath(algo=self.args.house_alg +"_"+ self.args.gov_alg, n=self.args.n_households, task=self.envs.gov_task)
        # 保存参数到文件
        save_args(path=self.model_path, args=self.args)
        # 训练指标初始化
        self.households_welfare = 0
        self.find_best_response = self.args.find_best_response
        self.eva_year_indicator = 0
        self.wandb = self.args.wandb

        # 是否使用Wandb标志
        if self.wandb:
            wandb.init(  # 初始化Wandb实验跟踪
                config=dict(self.args),
                project="MACRO",
                entity="ai_tax",
                name=self.model_path.parent.parent.parent.name + "_" + self.model_path.name + '_' + str(self.args.n_households)+"_"+self.envs.gov_task+"_seed="+str(self.args.seed),
                dir=str(self.model_path),
                job_type="training",
                reinit=True
            )
    
    def _get_tensor_inputs(self, obs):
        """将观察值转换为Tensor
                Args:
                    obs: 原始观察值(numpy数组)
                Returns:
                    obs_tensor: 转换为指定设备的Tensor
                """

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        return obs_tensor
        
    def run(self):
        # 智能体列表
        agents = [self.government_agent, self.house_agent]
        # 训练指标记录列表
        gov_rew, house_rew, epochs = [], [], []
        # 环境初始化
        global_obs, private_obs = self.envs.reset()
        # 重新初始化归一化模块
        self.global_state_norm = Normalization(shape=self.envs.government.observation_space.shape[0])
        self.private_state_norm = Normalization(shape=self.envs.households.observation_space.shape[0])
        global_obs = self.global_state_norm(global_obs)
        private_obs = self.private_state_norm(private_obs)

        # 训练频率记录（未使用）
        train_freq = []
        # 开始训练循环
        for epoch in range(self.args.n_epochs):
            transition_dict = {'global_obs': [], 'private_obs': [], 'gov_action': [], 'house_action': [],'gov_reward': [],
                               'house_reward': [], 'next_global_obs': [], 'next_private_obs': [], 'done': [], "mean_house_actions": []}
            sum_loss = np.zeros((len(agents), 2))

            # 经验收集阶段
            for t in range(self.args.epoch_length):
                # 获取状态
                global_obs_tensor = self._get_tensor_inputs(global_obs)
                private_obs_tensor = self._get_tensor_inputs(private_obs)
                # 政府智能体选择动作
                gov_action = self.government_agent.get_action(global_obs_tensor=global_obs_tensor,
                                                              private_obs_tensor=private_obs_tensor,
                                                              agent_name="government")
                # 家庭智能体选择动作
                house_action = self.house_agent.get_action(global_obs_tensor=global_obs_tensor,
                                                           private_obs_tensor=private_obs_tensor,
                                                           gov_action=gov_action, agent_name="household")
                # 处理mean-field算法的平均动作
                if "mf" in self.args.house_alg:
                    house_action, mean_house_action = house_action
                else:
                    mean_house_action = None
                # 构建环境动作字典
                action = {self.envs.government.name: gov_action,
                          self.envs.households.name: house_action}
                # 环境执行一步
                next_global_obs, next_private_obs, gov_reward, house_reward, done = self.envs.step(action)
                # 归一化下一观察
                next_global_obs = self.global_state_norm(next_global_obs)
                next_private_obs = self.private_state_norm(next_private_obs)

                # 存储经验数据
                if agents[0].on_policy or agents[1].on_policy:  # on-policy算法使用transition_dict
                    # on policy
                    transition_dict['global_obs'].append(global_obs)
                    transition_dict['private_obs'].append(private_obs)
                    transition_dict['gov_action'].append(gov_action)
                    transition_dict['house_action'].append(house_action)
                    transition_dict['gov_reward'].append(gov_reward)
                    transition_dict['house_reward'].append(house_reward)
                    transition_dict['next_global_obs'].append(next_global_obs)
                    transition_dict['next_private_obs'].append(next_private_obs)
                    transition_dict['done'].append(float(done))
                    transition_dict['mean_house_actions'].append(mean_house_action)
                if (not agents[0].on_policy) or (not agents[1].on_policy):  # off-policy算法使用buffer
                    # off policy: replay buffer
                    self.buffer.add(global_obs, private_obs, gov_action, house_action, gov_reward, house_reward,
                                    next_global_obs, next_private_obs, float(done), mean_action=mean_house_action)
                # 更新观察
                global_obs = next_global_obs
                private_obs = next_private_obs
                # 环境终止处理
                if done:
                    global_obs, private_obs = self.envs.reset()
                    global_obs = self.global_state_norm(global_obs)
                    private_obs = self.private_state_norm(private_obs)

            # for leader, if follower is not BR, break
            # 策略更新阶段
            for i in range(len(agents)):
                # if epoch < 10 or epoch % 5 == 0 or i == 1:
                if agents[i].on_policy == True:  # on-policy算法训练
                    actor_loss, critic_loss = agents[i].train(transition_dict)
                    sum_loss[i, 0] = actor_loss
                    sum_loss[i, 1] = critic_loss
                else:  # off-policy算法训练
                    for _ in range(self.args.update_cycles):
                        transitions = self.buffer.sample(self.args.batch_size)
                        actor_loss, critic_loss = agents[i].train(transitions, other_agent=agents[1-i])  # MARL has other agents
                        sum_loss[i, 0] += actor_loss
                        sum_loss[i, 1] += critic_loss
                    
            # print the log information  # 日志记录和模型保存
            if epoch % self.args.display_interval == 0:
                if epoch == 20:
                    write_flag = False
                else:
                    write_flag = False
                # 评估当前策略
                economic_idicators_dict = self._evaluate_agent(write_evaluate_data=write_flag)
                # 保存指标数据
                now_step = (epoch + 1) * self.args.epoch_length
                gov_rew.append(economic_idicators_dict["gov_reward"])
                house_rew.append(economic_idicators_dict["house_reward"])
                np.savetxt(str(self.model_path) + "/gov_reward.txt", gov_rew)
                np.savetxt(str(self.model_path) + "/house_reward.txt", house_rew)
                epochs.append(now_step)
                np.savetxt(str(self.model_path) + "/steps.txt", epochs)
                # 计算损失指标
                loss_dict = {
                    "house_actor_loss": sum_loss[1, 0],
                    "house_critic_loss": sum_loss[1, 1],
                    "gov_actor_loss": sum_loss[0, 0],
                    "gov_critic_loss": sum_loss[0, 1]
                }
                # 计算可利用性指标
                if self.find_best_response == True:
                    exploitability_rate = self.judge_best_response()
                else:
                    exploitability_rate = 100
                exploitability_dict = {
                    "exploitability": exploitability_rate
                }
                # 记录到Wandb
                if self.wandb:
                    wandb.log(economic_idicators_dict)
                    wandb.log(loss_dict)
                    wandb.log(exploitability_dict)
                # 控制台输出
                print(
                    '[{}] Epoch: {} / {}, Frames: {}, Gov_Rewards: {:.3f}, House_Rewards: {:.3f}, years:{:.3f}, actor_loss: {:.3f}, critic_loss: {:.3f}, exploitability_rate: {:.6f}'.format(
                        datetime.now(), epoch, self.args.n_epochs, (epoch + 1) * self.args.epoch_length,
                        economic_idicators_dict["gov_reward"], economic_idicators_dict["house_reward"],
                        economic_idicators_dict["years"], np.sum(sum_loss[:,0]), np.sum(sum_loss[:,1]), exploitability_rate))
            # 定期保存模型
            if epoch % self.args.save_interval == 0:
                self.house_agent.save(dir_path=self.model_path)
                self.government_agent.save(dir_path=self.model_path)
        # 训练结束处理
        if self.wandb:
            wandb.finish()

    def test(self,house_model_path, government_model_path):
        """测试训练好的模型
               Args:
                   house_model_path: 家庭智能体模型路径
                   government_model_path: 政府智能体模型路径
               """
        # 加载训练好的模型
        if self.args.house_alg != "real" and self.args.house_alg != "rule_based":
            self.house_agent.load(dir_path=house_model_path)
        if self.args.gov_alg != "rule_based" and self.args.gov_alg != "saez" and self.args.gov_alg != "us_federal":
            self.government_agent.load(dir_path=government_model_path)
        # 进行评估
        economic_idicators_dict = self._evaluate_agent(write_evaluate_data=True)
        print(economic_idicators_dict)
    
    def test_heter_agents(self,house_model_path, government_model_path, multiple_heter_agents_numbers):
        """测试异构智能体组合
                Args:
                    house_model_path: 基础家庭智能体路径
                    government_model_path: 政府智能体路径
                    multiple_heter_agents_numbers: 异构智能体数量配置列表
        """

        # 加载模型
        # n_households: 100
        # self.args.house_alg = bi_mfrl
        # house_model_path = agents / models / bi_mfrl_bi_ddpg / 100 / gdp / run129 / bimf_house_actor.pt
        # self.args.gov_alg = ddpg
        # government_model_path = agents / models / mfrl_ddpg / 100 / gdp / run15 / ddpg_net.pt

        # multiple_heter_agents_numbers = [[100, 100, 100, 100],
        #                                  [100, 100, 100, 100],
        #                                  [70, 80, 90, 100],
        #                                  [40, 60, 80, 100],
        #                                  [25, 50, 75, 100]]
        if self.args.house_alg != "real" and self.args.house_alg != "rule_based":
            self.house_agent.load(dir_path=house_model_path)
        if self.args.gov_alg != "rule_based" and self.args.gov_alg != "saez" and self.args.gov_alg != "us_federal":
            self.government_agent.load(dir_path=government_model_path)
        # 对不同异构配置进行评估s
        economic_idicators_list = []
        for each_heter_number in multiple_heter_agents_numbers:
            economic_idicators = self._evaluate_agent(heter_house_n=each_heter_number)
            economic_idicators_list.append(economic_idicators)
        # 返回平均指标
        return dict(zip(['Your_agent_average_utility', 'Your_agent_utility_relative_to_all', 'Government_return'], np.mean(economic_idicators_list,axis=0)))

    def init_economic_dict(self, gov_reward, households_reward):
        """初始化经济指标字典
                Args:
                    gov_reward: 政府奖励
                    households_reward: 家庭奖励列表
        """
        self.econ_dict = {
            "gov_reward": gov_reward,  # 政府累计奖励
            "social_welfare": np.sum(households_reward),  # 社会福利（总家庭奖励）
            "house_reward": households_reward,  # 各家庭奖励列表
            "years": getattr(self.eval_env, 'step_cnt', None),  # 模拟年数
            "house_income": getattr(self.eval_env, 'post_income', None),  # 家庭税后收入
            "house_total_tax": getattr(self.eval_env, 'tax_array', None),  # 家庭总税收
            "house_income_tax": getattr(self.eval_env, 'income_tax', None),  # 家庭所得税
            "house_wealth": getattr(self.eval_env, 'households', None).at_next if hasattr(self.eval_env,
                                                                                          'households') else None,
        # 家庭财富
            "house_wealth_tax": getattr(self.eval_env, 'asset_tax', None),  # 财富税
            "per_gdp": getattr(self.eval_env, 'per_household_gdp', None),  # 人均GDP
            "GDP": getattr(self.eval_env, 'GDP', None),  # 国内生产总值
            "income_gini": getattr(self.eval_env, 'income_gini', None),  # 收入基尼系数
            "wealth_gini": getattr(self.eval_env, 'wealth_gini', None),  # 财富基尼系数
            "WageRate": getattr(self.eval_env, 'WageRate', None),  # 工资率
            "total_labor": getattr(self.eval_env, 'Lt', None),  # 总劳动力
            "house_consumption": getattr(self.eval_env, 'consumption', None),  # 家庭消费
            "house_work_hours": getattr(self.eval_env, 'ht', None),  # 工作时长
            "gov_spending": (getattr(self.eval_env, 'Gt_prob', None) * getattr(self.eval_env, 'GDP', None))
            if hasattr(self.eval_env, 'Gt_prob') and hasattr(self.eval_env, 'GDP') else None  # 政府支出
        }

    def judge_best_response(self, transition_dict=None):
        # fix the leader's policy, update follower's policy until follower's value don't change.
        """判断是否达到最佳响应
                Args:
                    transition_dict: 经验过渡字典（当使用on-policy时）
                Returns:
                    exploitability_rate: 可利用性比率
                """
        # 计算当前指标
        current_indicators = self._evaluate_agent()
        current_households_welfare = current_indicators["social_welfare"]
        current_government_payoff = current_indicators['gov_reward']
        # 创建智能体副本进行策略更新测试
        house_agent_update = copy.copy(self.house_agent)
        government_agent_update = copy.copy(self.government_agent)
        # 执行策略更新
        if self.house_agent.on_policy == True:
            actor_loss, critic_loss = house_agent_update.train(transition_dict)
            actor_loss, critic_loss = government_agent_update.train(transition_dict)
        else:
            for _ in range(self.args.update_cycles):
                transitions = self.buffer.sample(self.args.batch_size)
                house_agent_update.train(transitions, other_agent=self.government_agent)  # MARL has other agents
                government_agent_update.train(transitions, other_agent=self.house_agent)  # MARL has other agents
        # 计算新指标
        new_households_welfare = self._evaluate_agent(single_update_household=house_agent_update, judge_exploitability=True)["social_welfare"]
        new_government_payoff = self._evaluate_agent(single_update_government=government_agent_update, judge_exploitability=True)["gov_reward"]
        
        exploitability_rate = abs((new_households_welfare - current_households_welfare)/current_households_welfare) + abs((new_government_payoff - current_government_payoff)/current_government_payoff)
        return exploitability_rate

    # economic_idicators = self._evaluate_agent(heter_house_n=each_heter_number)
    # 评估智能体方法，用于在评估环境中运行智能体并记录经济指标
    def _evaluate_agent(self, single_update_household=None, single_update_government=None, judge_exploitability=False, write_evaluate_data=False,heter_house_n=None):
        """评估智能体性能
                Args:
                    single_update_household: 单个更新的家庭智能体（用于最佳响应判断）
                    single_update_government: 单个更新的政府智能体
                    judge_exploitability: 是否判断可利用性
                    write_evaluate_data: 是否写入评估数据
                    heter_house_n: 异构家庭智能体数量配置
                Returns:
                    经济指标字典或评估值列表
                """
        # 异构智能体数量：100
        heter_house_number = [self.args.n_households]
        # eval_econ = ["gov_reward", "house_reward", "social_welfare", "per_gdp","income_gini","wealth_gini","years","GDP"]
        # 初始化评估指标存储
        eval_econ = ["gov_reward", "house_reward", "social_welfare", "per_gdp", "income_gini",
                     "wealth_gini", "years", "GDP", "gov_spending", "house_total_tax", "house_income_tax",
                     "house_wealth_tax", "house_wealth", "house_income", "house_consumption", "house_work_hours",
                     "total_labor", "WageRate"]
        eval_values = []

        episode_econ_dict = dict(zip(eval_econ, [[] for i in range(len(eval_econ))]))
        final_econ_dict = dict(zip(eval_econ, [None for i in range(len(eval_econ))]))

        # 多轮评估取平均 self.args.eval_episodes=5
        for epoch_i in range(self.args.eval_episodes):
            # 重置评估环境
            global_obs, private_obs = self.eval_env.reset()

            print('global_obs=',global_obs)
            print('private_obs=',private_obs)
            print('=='*1000)
            # 初始化全局和私有状态归一化
            self.global_state_norm = Normalization(shape=self.envs.government.observation_space.shape[0])
            self.private_state_norm = Normalization(shape=self.envs.households.observation_space.shape[0])
            global_obs = self.global_state_norm(global_obs)
            private_obs = self.private_state_norm(private_obs)

            # 初始化本轮指标存储
            eval_econ_dict = dict(zip(eval_econ, [[] for i in range(len(eval_econ))]))



            self.init_economic_dict(0, 0)
            # 单轮评估循环
            while True:
                with torch.no_grad():  # 禁用梯度计算
                    # 获取当前观察的张量形式
                    global_obs_tensor = self._get_tensor_inputs(global_obs)
                    private_obs_tensor = self._get_tensor_inputs(private_obs)
                    # 政府动作选择
                    if judge_exploitability == True and single_update_government != None:
                        gov_action = single_update_government.get_action(global_obs_tensor=global_obs_tensor,
                                                                      private_obs_tensor=private_obs_tensor,
                                                                      agent_name="government")
                    else:
                        # gov_action = [-0.99215483 - 1.00510138  1.00043424 - 1.001328    0.58209762]
                        # len(gov_action) = 5
                        gov_action = self.government_agent.get_action(global_obs_tensor=global_obs_tensor,
                                                                  private_obs_tensor=private_obs_tensor,
                                                                  agent_name="government")

                    # 家庭动作选择
                    house_action = self.house_agent.get_action(global_obs_tensor=global_obs_tensor,
                                                               private_obs_tensor=private_obs_tensor,
                                                               gov_action=gov_action, agent_name="household")
                    # array([[0.75099001, 0.91538004],
                    #        [0.74865115, 0.90710963],
                    #        ...,
                    #        [0.74805519, 0.92897303]]),
                    # array([0.75022812, 0.93637805]))
                    # house_action里面有两个数组，第一个形状为（100，2），第二个是（2，）



                    if "mf" in self.args.house_alg:
                        house_action, _ = house_action

                    # 异构智能体处理 heterogeneous_house_agent = True
                    if self.args.heterogeneous_house_agent:
                        heter_house_number = heter_house_n  #
                        #   [ [100, 100, 100, 100],
                        #     [100, 100, 100, 100],
                        #     [70, 80, 90, 100],
                        #     [40, 60, 80, 100],
                        #     [25, 50, 75, 100]]
                        # 分割不同策略的动作
                        your_policy_action = house_action[:heter_house_number[0]]
                        heter_action_list = [your_policy_action]
                        # 获取异构智能体动作
                        for i, each_heter_agent in enumerate(self.heter_house):
                            each_heter_action = each_heter_agent.get_action(global_obs_tensor=global_obs_tensor, private_obs_tensor=private_obs_tensor, gov_action=gov_action, agent_name="household")
                            if isinstance(each_heter_action, tuple):
                                each_heter_action = each_heter_action[0]
                            heter_action_list.append(each_heter_action[heter_house_number[i]:heter_house_number[i+1]])
                        house_action = np.vstack(heter_action_list)
                    # 最佳响应测试时的动作替换
                    if judge_exploitability == True and single_update_household != None:

                        update_house_action = single_update_household.get_action(global_obs_tensor=global_obs_tensor,
                                                                                 private_obs_tensor=private_obs_tensor,
                                                                                 gov_action=gov_action,
                                                                                 agent_name="household")

                    if judge_exploitability == True and single_update_household != None:
                        if "mf" in self.args.house_alg:
                            update_house_action, _ = update_house_action
                        # 随机替换一个家庭的动作
                        random_house_index = np.random.randint(0, self.args.n_households)
                        house_action[random_house_index] = update_house_action[random_house_index]
                        if "mf" in self.args.house_alg:
                            mean_house_action = np.mean(house_action, axis=-2)

                    # 执行环境步进
                    action = {self.envs.government.name: gov_action,
                              self.envs.households.name: house_action}


                    next_global_obs, next_private_obs, gov_reward, house_reward, done = self.eval_env.step(action)
                    # 归一化下一观察
                    next_global_obs = self.global_state_norm(next_global_obs)
                    next_private_obs = self.private_state_norm(next_private_obs)
                # 记录经济指标
                self.init_economic_dict(gov_reward, house_reward)
                for each in eval_econ:
                    if "house_" in each:
                        # 家庭相关指标转换为列表存储
                        eval_econ_dict[each].append(self.econ_dict[each].tolist())
                    else:
                        eval_econ_dict[each].append(self.econ_dict[each])
                # 环境终止判断
                if done:
                    # print(self.eval_env.step_cnt)
                    break
                # 更新观察
                global_obs = next_global_obs
                private_obs = next_private_obs
            # 处理本轮评估结果
            for key, value in eval_econ_dict.items():
                if key == "gov_reward" or key == "house_reward" or key == "GDP":
                    # 累计型指标求和
                    episode_econ_dict[key].append(np.sum(value))
                elif key == "years":
                    # 年份取最大值
                    episode_econ_dict[key].append(np.max(value))
                else:
                    # 其他指标取平均
                    episode_econ_dict[key].append(np.mean(value))
            # 计算特定指标
            your_agent_utility = np.mean(np.sum(eval_econ_dict["house_reward"],axis=0)[:heter_house_number[0]])
            your_utility_vs_all = your_agent_utility / np.mean(np.sum(eval_econ_dict["house_reward"],axis=0))
            gov_reward = np.sum(eval_econ_dict["gov_reward"])
            # 控制台输出进度
            print(
                'Gov_return: {:.3f}, Your House_Rewards: {:.3f}, Your House_Rewards / all agents: {:.3f}, years: {:.3f}, Total GDP: {:.1f}, Total social welfare: {:.1f}'.format(
                    gov_reward,your_agent_utility, your_utility_vs_all, eval_econ_dict["years"][-1],
                    np.sum(eval_econ_dict['per_gdp']), np.sum(eval_econ_dict['social_welfare'])))
            eval_values.append([your_agent_utility, your_utility_vs_all, gov_reward])
        # 最终指标计算
        for key, value in episode_econ_dict.items():
            final_econ_dict[key] = np.mean(value)
        # 将字典直接写入文件  # 数据存储判断
        if self.econ_dict['years'] >= self.eva_year_indicator:
            write_evaluate_data = write_evaluate_data and True
            self.eva_year_indicator = self.econ_dict['years']
        # 写入评估数据
        if write_evaluate_data == True:
            print("============= Finish ================")
            print("============= Finish ================")

            store_path = "agents/data/"+self.args.economic_shock+"/N="+str(self.args.n_households)+"/"
            if not os.path.exists(os.path.dirname(store_path)):
                os.makedirs(os.path.dirname(store_path))
            with open(store_path + self.args.house_alg + "_" + self.args.gov_alg + "_" + str(
                    self.args.n_households) + "_data.json", "w") as file:
                json.dump(eval_econ_dict, file, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        # 返回结果
        if self.args.heterogeneous_house_agent:
            return np.mean(eval_values, axis=0)
        else:
            return final_econ_dict


def save_args(path, args):
    argsDict = args.__dict__
    with open(str(path) + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)
    
    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x
    
    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)


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


import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from singleagent.multi_discrete import MultiDiscrete
import random
# import sys
# sys.path.append(r'E:/MEC_maddpg_lstm/multiagent')
# from scenarios.simple import Scenario


# environment for all agents in the multiagent world
# 所有智能体的环境
# currently code assumes that no agents will be created/destroyed at runtime!
# 当前代码假定在运行时不会创建/销毁任何智能体！
class AgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None, done_callback=None, shared_viewer=True):
        # world, reset_callback=None, reward_callback=None, observation_callback=None 由world = scenario.make_world()传入
        # self.battery = battery
        self.world = world
        # self.users = self.world.users
        # self.services = self.world.services
        # self.time_long_term = None
        # self.world = world
        self.agent = self.world.agent
        # self.processing_delay = 0.
        # set required vectorized gym env property
        # 设置所需的矢量化gym环境属性
        # self.n = len(world.agents)
        # self.alpha = []
        # self.e_list = []
        self.max_service_type = world.max_service_type

        # scenario callbacks
        # 场景回调
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback

        # environment parameters
        self.discrete_action_space = False
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = True
        # if true, every agent has the same reward
        self.shared_reward = False
        self.time = 0

        # configure spaces
        # 配置空间
        self.action_space = []
        self.observation_space = []
        # self.time = []
        total_action_space = []
        # """
        # 卸载动作空间
        offloading_act_space = spaces.Box(low=0, high=1, shape=(self.agent.num_UEs,),
                                          dtype=np.float32)
        total_action_space.append(offloading_act_space)

        # 关联动作空间
        association_act_space = spaces.Box(low=0, high=self.agent.num_servers + 1, shape=(self.agent.num_UEs,),
                                          dtype=np.float32)
        total_action_space.append(association_act_space)

        # 缓存动作空间， 取值为0到1的浮点数
        caching_act_space = spaces.Box(low=0, high=1, shape=(self.agent.num_servers, self.max_service_type),
                                       dtype=np.float32)
        total_action_space.append(caching_act_space)

        # power allocation action space
        # 计算资源分配动作空间，即alpha，取值为0到1的浮点数
        comp_act_space = spaces.Box(low=0, high=1, shape=(self.agent.num_servers, self.agent.num_UEs), dtype=np.float32)
        total_action_space.append(comp_act_space)

        # bandwidth allocation action space
        # 带宽资源分配动作空间， 即beta，取值为0到1的浮点数
        band_act_space = spaces.Box(low=0, high=1, shape=(self.agent.num_servers, self.agent.num_UEs), dtype=np.float32)
        total_action_space.append(band_act_space)

        if len(total_action_space) > 1:  # len=4
            if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
            else:
                act_space = spaces.Tuple(total_action_space)
            self.action_space.append(act_space)
        else:
            self.action_space.append(total_action_space[0])

        # observation space
        obs_dim = len(observation_callback(self.agent, self.world))
        # array([24., 23., 17., 22., 15.,  1.,  2.,  4.,  2.,  3.])
        # self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
        self.observation_space.append(spaces.Discrete(obs_dim))
        # self.time = []
        # rendering
        self.shared_viewer = shared_viewer
        # if self.shared_viewer:
        #     self.viewers = [None]
        # else:
        #     self.viewers = [None] * self.n
        # self._reset_render()
        # """

    def step(self, action):
        obs_n = []
        reward_n = []
        done_n = []
        # info_n = {'n': []}
        act_n = []
        hit_n = []
        delay_n = []
        # hit_local_bs = []
        # set action for each agent
        # 为每个智能体设置动作
        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
        act_n.append(self._set_action(action[0], self.agent, self.action_space[0]))
        # 更新WORLD状态
        self.world.step()
        # record observation for each agent
        obs_n.append(self._get_obs(self.agent))
        reward_n.append(self._get_reward(self.agent))
        hit_n.append(self._get_num_hit(self.agent))
        delay_n.append(self._get_time_delay(self.agent))
        done_n.append(self._get_done(self.agent))
        reward = np.sum(reward_n)
        cost = self._get_cost(self.agent)

        # if self.shared_reward:
        #     reward_n = [reward] * self.n

        return obs_n, reward, done_n, act_n, delay_n, hit_n, cost

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        obs_n.append(self._get_obs(self.agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    # 观察特定agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)[0]

    def _get_cost(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)[3]

    def _get_num_hit(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)[2]

    def _get_num_local_hit(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)[2]

    def _get_time_delay(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)[1]

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        # print(action)
        # print(action_space)
        if isinstance(action_space, MultiDiscrete):  # 判断是否是多维离散空间
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:  # size
                act.append(action[index:(index + s)])  # 将传进来的action赋值给act,action的大小是space大小？
                index += s
            action = act
        else:
            action = action
        map_offloading_action = (action[: agent.num_UEs] + 1) / 2

        map_association_action = np.round((((action[agent.num_UEs:agent.num_UEs*2] + 1) * agent.num_servers) / 2) + 0)
        # map_caching_action为0或者1的数，0没有服务，1有服务
        map_caching_action = np.round((((action[agent.num_UEs*2:agent.num_UEs*2 + agent.num_servers * self.max_service_type] + 1) * 1) / 2) + 0)

        # aa = self.agent.num_UEs + self.agent.num_servers * self.max_service_type
        # alpha = (action[aa: aa + self.agent.num_servers * self.agent.num_UEs] + 1.0001) / 2
        #
        # bb = self.agent.num_UEs + self.agent.num_servers * self.max_service_type + self.agent.num_servers * self.agent.num_UEs
        # beta = (action[bb:] + 1.0001) / 2

        # map_comp_power_action计算资源分配，总和为一
        # map_bandwidth_action带宽资源分配，总和为一

        alpha_old = action[agent.num_UEs*2 + agent.num_servers * self.max_service_type:agent.num_UEs*2 + agent.num_servers * self.max_service_type + agent.num_servers * agent.num_UEs]
        beta_old = action[agent.num_UEs*2 + agent.num_servers * self.max_service_type + agent.num_servers * agent.num_UEs:]
        alpha_new = []
        beta_new = []
        for j in range(agent.num_servers):
            alpha_sum = sum(alpha_old[j * agent.num_UEs:(j + 1) * agent.num_UEs] + 1.0001)
            beta_sum = sum(beta_old[j * agent.num_UEs:(j + 1) * agent.num_UEs] + 1.0001)
            alpha = (alpha_old[j * agent.num_UEs:(j + 1) * agent.num_UEs] + 1.0001) / alpha_sum
            beta = (beta_old[j * agent.num_UEs:(j + 1) * agent.num_UEs] + 1.0001) / beta_sum
            # print(sum(alpha))
            alpha_new.append(alpha)
            beta_new.append(beta)
        # print(alpha_new)

        alpha = np.concatenate(alpha_new, axis=0)
        # print(alpha)
        beta = np.concatenate(beta_new, axis=0)

        map_bandwidth_action = alpha
        map_comp_power_action = beta
        # print(alpha)

        agent.action.offloading = map_offloading_action
        agent.action.association = map_association_action
        agent.action.caching = map_caching_action
        agent.action.trans_band = map_bandwidth_action
        agent.action.trans_power = map_comp_power_action

        return agent.action

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    def render(self, mode='human'):
        if mode == 'human':
            pass
        return None

    def close(self):
        return None


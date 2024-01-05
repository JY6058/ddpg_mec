import numpy as np
import random
from myUnits import random_pick_size, random_pick_service
import math
# from scenarios.simple import Scenario
from scipy import stats
# import sys
# sys.path.append(r'E:/MEC_maddpg_lstm')
import service_produced as sp
import runner
from sklearn import preprocessing


class AgentState(object):
    def __init__(self):
        # self.caching_storage = None  # 需要？
        # self.tasks_size = None
        self.requested_service = None
        self.n_task = None
        self.delay_tolerance = None
        # self.adj = []  # 自己定义
        # self.e = None


class Action(object):
    def __init__(self):
        self.offloading = None
        self.association = None
        self.caching = None
        self.trans_band = None
        self.trans_power = None


class Agent(object): # agent基站,设置基站属性
    def __init__(self):
        self.name = ''
        self.serial_number = None
        # self.bandwidth = 200 # 20.0 # 单位MHz, 边缘服务器的带宽资源 10MHz
        # self.comp = 12500 # 3200  # 单位MHz, 边缘服务器的计算资源3200MHz
        self.bandwidth = 200  # M
        self.comp = [15000, 15000, 15000]
        self.cache_storage = 50 # 200 # 单位200GB, 缘服务器中的缓存资源
        # self.each_storage = 10 # 10  # 每个service所需存储空间GB
        self.trans_BSs = 1.8 # 2 # 单位Mbps, 边缘服务器的转发到其他基站的速率
        self.trans_cloud = 0.5 # 0.5 # 单位Mbps, 边缘服务器的转发到云端的速率
        # self.gain = 100 ** (-3) # 信道增益-6dBm
        self.path_loss_factor = 3
        self.bias = 0.5  # DT偏差
        # self.alpha = [] # 频率分配系数
        # self.beta = [] # 带宽分配系数
        # self.exist = []
        # self.kind = []
        self.num_UEs = 15  # 一个基站下有10个用户
        self.num_servers = 3
        self.state = AgentState()  # 资源、任务、缓存
        # self.delay_requirements = 1 # 每个用户时延需求一样 单位 s/Mbit
        self.action = Action()  # 卸载、计算、频谱、缓存
        self.sigma = 1e-8  # 单位-104dBm
        self.interference = 1e-8  # 其他用户的干扰

        self.delta = None  # 流行度参数
        self.pop_profile_vector = None
        self.services = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # self.n_delay = 10 # # 每个用户时延需求一样 单位 s/Mbit
        self.n_X = 737.5 # 每bit任务所需要的CPU,每个用户设为一样 cycles/Mbit
        self.n_l = 10 # 每个service所需存储空间GB
        self.comp_fre = 800 # 800MHz
        self.power = 0.5  # 用户发送功率5W
        self.k = 1e-28  # 有效电容系数
        # 圆心坐标
        self.center_x = 0
        self.center_y = 0
        # 圆半径
        self.radius = 100

# multi-agent world
class World(object):
    # sigma = 4e-8 # 单位-104dBm
    # interference = np.random.uniform(0, 1)  # 其他用户的干扰
    def __init__(self):
        # 智能体和用户的个数
        # self.cloud = 1
        self.agent = Agent()
        # self.users = []
        self.max_service_type = 10  # service类型的数量
        # self.sigma = 1e-8 # 单位-104dBm
        # self.interference = 1e-8 # 其他用户的干扰
        self.eta = 5  # 超过BS缓存空间时所获得的惩罚 针对基站
        self.points = self.init_points()

    def init_points(self):
        # 记录所有点的坐标
        points = []
        # 生成用户随机坐标
        for i in range(self.agent.num_UEs):
            # 随机生成角度
            angle = random.uniform(0, 2 * math.pi)
            # 随机生成距离
            distance = random.uniform(0, self.agent.radius)
            # 计算坐标
            x = self.agent.center_x + distance * math.cos(angle)
            y = self.agent.center_y + distance * math.sin(angle)
            # 记录坐标
            points.append((x, y, "User"))
        # 生成SBS随机坐标

        angle_interval = 2 * math.pi / self.agent.num_servers
        for i in range(self.agent.num_servers):
            # 随机生成角度
            angle = i * angle_interval
            # angle = random.uniform(0, 2 * math.pi)
            # 随机生成距离
            # distance = random.uniform(0, self.agent.radius)
            distance = random.uniform(self.agent.radius/4, self.agent.radius*3/4)
            # 计算坐标
            x = self.agent.center_x + distance * math.cos(angle)
            y = self.agent.center_y + distance * math.sin(angle)
            # 记录坐标
            points.append((x, y, "SBS"))
        # 生成MBS随机坐标
        points.append((self.agent.center_x, self.agent.center_y, "MBS"))
        return points

    # 更新world状态
    def step(self):
        self.update_agent_state(self.agent)  # battery, j)

    def update_agent_state(self, agent):  # battery, i): # 针对每一个agent

        generate_requested_service = np.zeros(agent.num_UEs)
        generate_task_size = np.zeros(agent.num_UEs)
        generate_delay_tolerance = np.zeros(agent.num_UEs)
        # generate_e = np.zeros(agent.num_UEs)

        # generate_delay = np.zeros(agent.num_UEs)
        # generate_X = np.zeros(agent.num_UEs)
        # generate_l = np.zeros(agent.num_UEs)
        # generate_e = np.zeros(agent.num_UEs)
        # generate_comp = np.zeros(agent.num_UEs)
        popular_profile_vector = agent.pop_profile_vector
        probability_vector = list(map(lambda x: x / sum(popular_profile_vector), popular_profile_vector))
        # probability_vector = np.ones(self.max_service_type) * (1 / self.max_service_type)
        for j in range(agent.num_UEs):
            # n1 = random.randint(0, 9)
            # n = stats.poisson.pmf(n1, 4)
            # n2 = np.random.uniform(1, 5)
            # n21 = stats.poisson.pmf(n2, 2.5)
            # generate_requested_service[j] = round(n * n1)  # 任务所需的服务种类
            # generate_task_size[j] = n2 * n21
            generate_requested_service[j] = random_pick_service(agent, probability_vector) # sp.service_kind[i][j][runner.iter_num]
            # generate_requested_service[j] = random.randint(0, 9) # 任务所需的服务种类
            generate_task_size[j] = random_pick_size(agent) # np.random.uniform(10, 30)
            generate_delay_tolerance[j] = generate_task_size[j]  # random_pick_delay(agent)

        agent.state.requested_service = generate_requested_service
        # print(generate_requested_service)
        # print(np.round((((agent.state.requested_service - 0) * (10 - 1)) / 1) + 0))
        agent.state.n_task = generate_task_size
        agent.state.delay_tolerance = generate_delay_tolerance






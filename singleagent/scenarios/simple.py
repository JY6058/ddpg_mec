from __future__ import division
import math

import numpy as np

import runner
from singleagent.core import World, Agent
from singleagent.scenario import BaseScenario
from myUnits import random_pick_size, random_pick_e, random_pick_service
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        world.agent.delta = 0.8  # 流行度系数
        self.points = world.points
        self.plot_points(self.points,world)
        # delta = [0.8, 1, 0.8, 1, 1.2]  # 每个服务器下流行度参数 [0.6, 0.8, 0.6, 0.8, 1.2]
        max_service_type = len(world.agent.services)
        rank = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 对[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]的排序
        # 每个BS流行度配置,后续根据每个基站的流行度生成任务 服从Zipf分布(获得每种服务在对应基站上的流行度)
        pop_profile_vector = []  # 每个元素对应[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]的popularity
        for j in range(max_service_type):
            popularity_j = (sum(list(map(lambda x: (1 / x) ** world.agent.delta, rank)))) / (rank[j] ** world.agent.delta)
            pop_profile_vector.append(popularity_j)

        world.agent.pop_profile_vector = pop_profile_vector
        # print(world.agent.pop_profile_vector)
        self.reset_world(world)
        return world
    def plot_points(self, points,world):
        # 假设points是一个包含所有点坐标和类型的列表，每个点表示为 (x, y, type)
        x_user = []
        y_user = []
        x_sbs = []
        y_sbs = []
        x_mbs = []
        y_mbs = []

        for point in points:
            x, y, node_type = point
            if node_type == "User":
                x_user.append(x)
                y_user.append(y)
            elif node_type == "SBS":
                x_sbs.append(x)
                y_sbs.append(y)
            elif node_type == "MBS":
                x_mbs.append(x)
                y_mbs.append(y)

        # 绘制散点图
        plt.scatter(x_user, y_user, color='r', marker='o', label='User')
        plt.scatter(x_sbs, y_sbs, color='g', marker='^', label='SBS')
        plt.scatter(x_mbs, y_mbs, color='b', marker='s', label='MBS')

        circle = Circle((0, 0), world.agent.radius, fill=False, color='k', linestyle='--')
        plt.gca().add_patch(circle)
        # 设置坐标轴范围和标签
        plt.xlim((-world.agent.radius, world.agent.radius))
        plt.ylim((-world.agent.radius, world.agent.radius))
        plt.xlabel('X(m)')
        plt.ylabel('Y(m)')

        # 添加图例
        plt.legend()
        # 保存图形到文件
        plt.savefig('figure.png')

    def reset_world(self, world):
        # 设置初始化状态
        generate_requested_service = np.zeros(world.agent.num_UEs)
        generate_task_size = np.zeros(world.agent.num_UEs)
        generate_delay_tolerance = np.zeros(world.agent.num_UEs)

        popular_profile_vector = world.agent.pop_profile_vector
        probability_vector = list(map(lambda x: x / sum(popular_profile_vector), popular_profile_vector))
        # probability_vector = np.ones(world.max_service_type) * (1 / world.max_service_type)  # 均匀分布1/21
        for i in range(world.agent.num_UEs):
            generate_requested_service[i] = random_pick_service(world.agent, probability_vector)  # sp.service_kind[j][i][runner.iter_num]
            generate_task_size[i] = random_pick_size(world.agent)  # np.random.uniform(1, 5)
            generate_delay_tolerance[i] = generate_task_size[i]  # random_pick_delay(agent)

        world.agent.state.requested_service = generate_requested_service
        world.agent.state.n_task = generate_task_size
        world.agent.state.delay_tolerance = generate_delay_tolerance

            # agent.state.e = generate_e

    def reward(self, agent, world):
        # 将归一化的状态空间变换为实际的状态空间
        # NewValue = int((((OldValue - OldMin) * NewRange) / OldRange) + NewMin)
        task_size = np.round((((agent.state.n_task - 0) * 20) / 1) + 0) + 10
        delay_tolerance = np.round((((agent.state.delay_tolerance - 0) * 20) / 1) + 0) + 10
        requested_service = np.round((((agent.state.requested_service - 0) * (world.max_service_type - 1)) / 1) + 0)
        # e = (((agent.state.e - 0) * 4) / 1 + 0)

        # 确定卸载模式，计算对应时延
        proc_delay, hit_num = self.get_proc_delay(agent, world)  # 当前基站下每个用户的计算时延[5,]

        cost_delay = np.zeros(agent.num_UEs)
        for i in range(agent.num_UEs):
            if proc_delay[i] == 1000 or delay_tolerance[i] - proc_delay[i] < 0:  # 没有卸载选项
                cost_delay[i] = 30  # 随便设一个较大的数
            else:
                cost_delay[i] = proc_delay[i]
        # reward = []
        # 奖励=-时延-惩罚
        # 电池电量不足时或者处理时间大于约束时间的的惩罚
        # 时延约束回报
        reward_delay = np.zeros(agent.num_UEs)
        for i in range(agent.num_UEs):
            if proc_delay[i] > delay_tolerance[i] or proc_delay[i] == 1000:
                reward_delay[i] = 0
                # agent.reward = 0 - processing_delay-1
            else:
                reward_delay[i] = 2
        # 存储容量约束回报
        caching_penalty = np.zeros(agent.num_servers)
        for j in range(agent.num_servers):
            new_caching = agent.action.caching[j * world.max_service_type:(j+1) * world.max_service_type]
            if sum(new_caching) * agent.n_l <= agent.cache_storage:
                caching_penalty[j] = 2
            else:
                caching_penalty[j] = 0
        # 总回报
        reward = 0 - sum(cost_delay) + sum(reward_delay) + sum(caching_penalty)
        time_delay = 0 - sum(cost_delay)
        return reward, time_delay, hit_num, cost_delay

    def observation(self, agent, world):
        requested_service = []
        task_size = []
        delay_tolerance = []

        requested_service.append(agent.state.requested_service)
        # print(requested_service)
        task_size.append(agent.state.n_task)
        delay_tolerance.append(agent.state.delay_tolerance)

        return np.concatenate(requested_service + task_size + delay_tolerance)  #  + e)

    def get_proc_delay(self, agent, world):
        # 将归一化的状态空间变换为实际的状态空间
        # NewValue = int((((OldValue - OldMin) * NewRange) / OldRange) + NewMin)
        requested_service = np.round((((agent.state.requested_service - 0) * (world.max_service_type - 1)) / 1) + 0)
        # requested_service = agent.state.requested_service
        task_size = np.round((((agent.state.n_task - 0) * 20) / 1) + 0) + 10
        delay_tolerance = np.round((((agent.state.delay_tolerance - 0) * 20) / 1) + 0) + 10
        # e = (((agent.state.e - 0) * 4) / 1 + 0)

        proc_delay = np.zeros(agent.num_UEs)  # 当前基站下每个用户的处理时延

        num_bs_processing = 0
        for i in range(agent.num_UEs):
            flag_local = 0
            flag_bs = 0
            for j in range(agent.num_servers):
                caching_j = agent.action.caching[j * world.max_service_type:(j * world.max_service_type + world.max_service_type)]
                if agent.action.association[i] == j and caching_j[int(requested_service[i])] == 1.0 and agent.action.trans_band[j*agent.num_UEs+i]>0:
                    a = agent.action.offloading[i]
                    proc_delay[i] = self.loc_time(i, agent, 1-a) + self.off_time(i, j, agent, a)
                    flag_bs = 1
                    num_bs_processing += 1
                elif agent.action.association[i] == j and caching_j[int(requested_service[i])] != 1.0:
                    proc_delay[i] = self.loc_time(i, agent, 1)
                    flag_local = 1

            if flag_local + flag_bs < 1:
                proc_delay[i] = 1000  # 不满足上述任何任务情况
            else:
                pass

        if 0.0 in proc_delay:
            assert False

        return proc_delay, num_bs_processing

    # 本地计算时间
    def loc_time(self, i, agent, a):
        # a 表示比例
        task_size = np.round((((agent.state.n_task - 0) * 20) / 1) + 0) + 10
        delay_tolerance = np.round((((agent.state.delay_tolerance - 0) * 20) / 1) + 0) + 10

        # self.loc_comp_time = task_size[i] * agent.n_X / (agent.comp_fre*(1+agent.bias))
        self.loc_comp_time = a * task_size[i] * agent.n_X / agent.comp_fre
        return self.loc_comp_time

    # 卸载时延
    def off_time(self, i, j, agent, a):
        task_size = np.round((((agent.state.n_task - 0) * 20) / 1) + 0) + 10
        delay_tolerance = np.round((((agent.state.delay_tolerance - 0) * 20) / 1) + 0) + 10
        alpha = agent.action.trans_band[j*agent.num_UEs+i]
        beta = agent.action.trans_power[j*agent.num_UEs+i]
        self.off_cmp_time = self.trans_time(i, j, alpha, agent, a) + self.bs_comp_time(i, j, beta, agent, a)

        return self.off_cmp_time

    # 传输时间
    def trans_time(self, i, j, alpha, agent, a):
        task_size = np.round((((agent.state.n_task - 0) * 20) / 1) + 0) + 10
        delay_tolerance = np.round((((agent.state.delay_tolerance - 0) * 20) / 1) + 0) + 10
        d = math.sqrt(pow(self.points[i][0]-self.points[agent.num_UEs+j][0], 2)+pow(self.points[i][1]-self.points[agent.num_UEs+j][1], 2))

        if alpha == 0:
            self.time_trans = 0
        else:
            self.trans_rate = alpha * agent.bandwidth * math.log2(1.0 + agent.power * d**(-agent.path_loss_factor) / (agent.sigma+agent.interference))
            self.time_trans = a * task_size[i] / self.trans_rate
        return self.time_trans

    # # MD到BS的传输能耗
    # def trans_energy(self, i, alpha, agent):
    #     task_size = np.round((((agent.state.n_task - 0) * 20) / 1) + 0) + 10
    #     delay_tolerance = np.round((((agent.state.delay_tolerance - 0) * 20) / 1) + 0) + 10
    #
    #     self.energy_trans = agent.power * self.trans_time(i, alpha, agent)
    #     return self.energy_trans
    #
    # # 本地计算能耗
    # def comp_energy(self, i, agent):
    #     task_size = np.round((((agent.state.n_task - 0) * 20) / 1) + 0) + 10
    #     delay_tolerance = np.round((((agent.state.delay_tolerance - 0) * 20) / 1) + 0) + 10
    #
    #     self.energy_comp = agent.k * task_size[i] * agent.n_X * agent.comp_fre ** 2
    #     return self.energy_comp

    # BS计算时间
    def bs_comp_time(self, i, j, beta, agent, a):
        task_size = np.round((((agent.state.n_task - 0) * 20) / 1) + 0) + 10
        delay_tolerance = np.round((((agent.state.delay_tolerance - 0) * 20) / 1) + 0) + 10

        # self.bs_time1 = task_size[i] * agent.n_X / beta / (agent.comp[j]*(1+agent.bias))
        self.bs_time1 = a * task_size[i] * agent.n_X / beta / agent.comp[j]
        return self.bs_time1

    # # BS转发任务到相邻BS的时间
    # def bs_bs_time(self, i, agent):
    #     task_size = np.round((((agent.state.n_task - 0) * 20) / 1) + 0) + 10
    #     delay_tolerance = np.round((((agent.state.delay_tolerance - 0) * 20) / 1) + 0) + 10
    #
    #     self.bs_trans_bs = task_size[i] / agent.trans_BSs
    #     return self.bs_trans_bs
    #
    # # bs转发到云端的计算时间
    # def bs_cloud_time(self, i, agent):
    #     task_size = np.round((((agent.state.n_task - 0) * 20) / 1) + 0) + 10
    #     delay_tolerance = np.round((((agent.state.delay_tolerance - 0) * 20) / 1) + 0) + 10
    #
    #     self.bs_cloud = task_size[i] / agent.trans_cloud
    #     return self.bs_cloud
    #
    # def battery(self, i, agent):
    #     e_max = 4  # 10J
    #     # print("agent.action.trans_band的大小")
    #     # print(agent.action.trans_band)
    #     alpha = agent.action.trans_band[i]
    #     e = (((agent.state.e-0) * 4) / 1 + 0)
    #     self.harvesting_e = np.random.uniform(0.05, 0.15)  # 收集的能量0-50mJ
    #     if 0 <= agent.action.offloading[i] < 2.5:
    #         self.next_e = min(max(e[i] - self.comp_energy(i, agent) + self.harvesting_e, 0), e_max)
    #     else:
    #         self.next_e = min(max(e[i] - self.trans_energy(i, alpha, agent) + self.harvesting_e, 0), e_max)
    #     # print("电量为 % d" % self.next_e)
    #     # self.next_e = preprocessing.MaxAbsScaler().fit_transform(np.array(self.next_e).reshape(-1, 1))
    #     return self.next_e


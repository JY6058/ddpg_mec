import random
import numpy as np
from sklearn import preprocessing
from pandas import DataFrame
import math
from random import choice
import matplotlib.pyplot as plt


def random_pick_size(agent):
    a = []
    for i in range(0, 20):  # (agent.min_task_size, agent.max_task_size + 1):
        b = []
        b.append(i)
        a.append(b)
    old_size = np.array(a)
    norm_size = preprocessing.MaxAbsScaler().fit_transform(old_size)

    probabilities = np.ones(21) * (1 / 21)  # 均匀分布
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(norm_size, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


def random_pick_e(agent):
    a = []
    for i in range(0, 5):
        b = []
        b.append(i)
        a.append(b)
    # print(a)
    old_e = np.array(a)
    # print(old_services)
    # 0 到 1 之间的归一化
    norm_e = preprocessing.MaxAbsScaler().fit_transform(old_e)
    # print(norm_services)

    probabilities = np.ones(5) * (1 / 5)  # 均匀分布
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(norm_e, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


def random_pick_service(agent, probabilities):
    a = []
    for i in range(0, len(agent.services)):  # (agent.min_task_size, agent.max_task_size + 1):
        b = []
        b.append(i)
        a.append(b)
    old_service = np.array(a)
    norm_service = preprocessing.MaxAbsScaler().fit_transform(old_service)

    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(norm_service, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item
# def randomwalk():
#     num_points = 100000
#     x0_values = [0]
#     x1_values = [0]
#     x2_values = [0]
#     x3_values = [0]
#     x4_values = [0]
#
#     while len(x0_values) < num_points:
#         a = np.random.uniform(0.1, 0.9)
#         x0_direction = np.random.choice([1, -1], p=[a, 1-a])
#         x0_step = x0_direction  # * x_distance
#         next_x0 = x0_values[-1] + x0_step
#         x0_values.append(next_x0)
#
#         b = np.random.uniform(0.1, 0.9)
#         x1_direction = np.random.choice([1, -1], p=[b, 1-b])
#         x1_step = x1_direction  # * x_distance
#         next_x1 = x1_values[-1] + x1_step
#         x1_values.append(next_x1)
#
#         c = np.random.uniform(0.1, 0.9)
#         x2_direction = np.random.choice([1, -1], p=[c, 1-c])
#         x2_step = x2_direction  # * x_distance
#         next_x2 = x2_values[-1] + x2_step
#         x2_values.append(next_x2)
#
#         d = np.random.uniform(0.1, 0.9)
#         x3_direction = np.random.choice([1, -1], p=[d, 1-d])
#         x3_step = x3_direction  # * x_distance
#         next_x3 = x3_values[-1] + x3_step
#         x3_values.append(next_x3)
#
#         e = np.random.uniform(0.1, 0.9)
#         x4_direction = np.random.choice([1, -1], p=[e, 1-e])
#         x4_step = x4_direction  # * x_distance
#         next_x4 = x4_values[-1] + x4_step
#         x4_values.append(next_x4)
#
#     min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 9))
#     # print(len(x0_values))
#     # print(len(x1_values))
#     # print(len(x2_values))
#     # print(len(x3_values))
#     # print(len(x4_values))
#
#     data00 = np.array(x0_values).reshape(-1, 1)
#     data00 = min_max_scaler.fit_transform(data00)
#     data00 = data00.reshape(1, -1)
#     data00 = data00.squeeze()
#
#     #
#     #     return x0_values #, x1_values, x2_values, x3_values, x4_values
#     #
#     # @staticmethod
#     # def normalize():
#     #     data0 = randomwalk
#     # # data0, data1, data2, data3, data4 = randomwalk()
#     # # 对利用随机游走模型生成的数据进行归一化处理，使其变成[0, 9]的数
#     #     min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 9))
#     #
#     #     data00 = np.array(data0).reshape(-1, 1)
#     #     data00 = min_max_scaler.fit_transform(data00)
#     #     data00 = data00.reshape(1, -1)
#     #     data00 = data00.squeeze()
#
#     data01 = np.array(x1_values).reshape(-1, 1)
#     data01 = min_max_scaler.fit_transform(data01)
#     data01 = data01.reshape(1, -1)
#     data01 = data01.squeeze()
#
#     data02 = np.array(x2_values).reshape(-1, 1)
#     data02 = min_max_scaler.fit_transform(data02)
#     data02 = data02.reshape(1, -1)
#     data02 = data02.squeeze()
#
#     data03 = np.array(x2_values).reshape(-1, 1)
#     data03 = min_max_scaler.fit_transform(data03)
#     data03 = data03.reshape(1, -1)
#     data03 = data03.squeeze()
#
#     data04 = np.array(x4_values).reshape(-1, 1)
#     data04 = min_max_scaler.fit_transform(data04)
#     data04 = data04.reshape(1, -1)
#     data04 = data04.squeeze()
#
#     # data00_r = []
#     # for i in range(len(data00)):
#     #     data00_r.append(round(data00[i]))
#     # plt.figure("原始0")
#     # plt.plot(range(num_points), x0_values)  # plt.plot(range(num_points))
#     # plt.figure("归一化0")
#     # plt.plot(range(num_points), data00)
#     # plt.figure("四舍五入0")
#     # plt.plot(range(num_points), data00_r)
#     #
#     # plt.figure("原始1")
#     # plt.plot(range(num_points), x1_values)  # plt.plot(range(num_points))
#     # plt.figure("归一化1")
#     # plt.plot(range(num_points), data01)
#     #
#     # plt.figure("原始2")
#     # plt.plot(range(num_points), x2_values)  # plt.plot(range(num_points))
#     # plt.figure("归一化2")
#     # plt.plot(range(num_points), data02)
#     #
#     # plt.figure("原始3")
#     # plt.plot(range(num_points), x3_values)  # plt.plot(range(num_points))
#     # plt.figure("归一化3")
#     # plt.plot(range(num_points), data03)
#     #
#     # plt.figure("原始4")
#     # plt.plot(range(num_points), x4_values)  # plt.plot(range(num_points))
#     # plt.figure("归一化4")
#     # plt.plot(range(num_points), data04)
#     #
#     # plt.show()
#
#     return [data00, data01, data02, data03, data04]

# a = randomwalk()

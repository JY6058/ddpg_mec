import random
import os
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim  # torch.optim包则主要包含了用来更新参数的优化算法，比如SGD、AdaGrad、RMSProp、 Adam
# import tensorflow as tf
from common.arguments import get_args


# define the actor network
class Actor(nn.Module):
    def __init__(self, args):
        # alpha：学习率
        # input_dims：输入维度
        # fc1_dims：第一层全连接层的神经元数量
        # fc2_dims：第二层全连接层的神经元数量
        # n_actions：动作数量
        # name：用于保存检查点文件的名称
        # chkpt_dir：检查点文件的目录，默认为’tmp/DDPG’
        super(Actor, self).__init__()

        self.max_action = args.high_action

        self.fc1 = nn.Linear(args.obs_shape, 256)  # 输入维度，输出维度
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(256)
        # 这段代码的作用是对ActorNetwork模型中的第一层全连接层进行初始化。其中，随机初始化有助于避免模型陷入局部最优解，而Batch Normalization则有助于加速训练过程，提高网络的收敛性能。

        self.fc2 = nn.Linear(256, 256)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(256)

        f3 = 0.003
        self.mu = nn.Linear(256, args.action_shape)
        # print(args.action_shape)
        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3, f3)

    def forward(self, state):
        # print(state.shape)
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.mu(x)
        # x = self.bn3(x)
        # print(x.shape)
        # print(T.tanh(x))
        x = self.max_action * T.tanh(x)
        # print(x)
        # x = T.tanh(self.mu(x))
        # print(x.squeeze(0))
        return x


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.max_action = args.high_action

        self.fc1 = nn.Linear(args.obs_shape+args.action_shape, 256)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(256)

        self.fc2 = nn.Linear(256, 256)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        self.bn2 = nn.LayerNorm(256)

        # 输出层
        # self.action_value = nn.Linear(self.fc2_dims, self.n_actions)
        # self.action_value = nn.Linear(args.action_shape, 128)
        f3 = 0.003
        self.q = nn.Linear(256, 1)
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)
        # self.bn3 = nn.LayerNorm(args.action_shape)

        # 构建Adam优化器self.optimizer，用于调整模型参数以最小化损失函数
        # 根据可用的设备（CPU或GPU）将模型放在相应的计算设备上
        # self.optimizer = optim.Adam(self.parameters(), lr=args.lr_critic)  # 用于调整模型参数以最小化损失函数

    def forward(self, state, action):
        # state = torch.cat(state, dim=1)
        # action = torch.cat(action, dim=1)
        # print(self.max_action)
        for i in range(len(action)):
            action[i] /= self.max_action
        # print(state)
        state = torch.cat([state, action], dim=1)
        # print(state)
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)  # weather this should appended
        # with a relu is an open debate

        # action_value = F.relu(self.action_value(action))
        # action_value = self.action_value(action)
        state_action_value = F.relu(state_value)
        state_action_value = self.q(state_action_value)
        # state_action_value = self.bn3(state_action_value)
        # print(state_action_value)

        return state_action_value


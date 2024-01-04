import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import tensorflow as tf


# define the actor network
class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        # self.offloading_max_action = args.offloading_high_action
        # self.offloading_min_action = args.offloading_low_action
        #
        # self.channel_max_action = args.channel_high_action
        # self.channel_min_action = args.channel_low_action
        #
        # self.caching_max_action = args.caching_high_action
        # self.caching_min_action = args.caching_low_action
        self.max_action = args.high_action
        # self.fc1 = nn.Conv1d(in_channels=args.obs_shape[agent_id], out_channels=75, kernel_size=1, groups=15, bias=True)
        self.fc1 = nn.Linear(args.obs_shape, 256)
        # self.lstm2 = nn.LSTM(256, 128, 1)
        self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(512, 256)
        self.action_out = nn.Linear(128, args.action_shape)  # shape 15

        # self.action_out1 = nn.Linear(128, 10)  # shape 10
        # self.action_out2 = nn.Linear(128, 15)  # shape 15

    def forward(self, x):
        # x = xavier_init(60, 128)
        # x = torch.nn.init.xavier_uniform_(x, gain=1)
        # x = x.unsqueeze(0)
        # x = x.permute(0, 2, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))
        # else:
        #     x = torch.split(x, [1280, 15], 1)
        #     x1 = x[0]
        #     x2 = x[1]
        #     x1 = F.relu(self.fc11(x1))
        #     x1, (h, c) = self.lstm2(x1)
        #     actions1 = self.max_action * torch.tanh(self.action_out1(x1)) # 服务缓存的动作
        #
        #     x2 = F.relu(self.fc1(x2))
        #     x2 = F.relu(self.fc3(x2))
        #     actions2 = self.max_action * torch.tanh(self.action_out2(x2)) # 卸载及资源分配的动作
        #
        #     actions = torch.cat((actions1, actions2), 1)
        # x = x.permute(0, 2, 1)
        # x, (h, c) = self.lstm2(x)
        # x = F.relu(self.fc3(x))
        # actions = self.max_action * F.relu(self.action_out(x))
        return actions


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        # self.offloading_max_action = args.offloading_high_action
        # self.offloading_min_action = args.offloading_low_action
        #
        # self.channel_max_action = args.channel_high_action
        # self.channel_min_action = args.channel_low_action
        #
        # self.caching_max_action = args.caching_high_action
        # self.caching_min_action = args.caching_low_action
        self.max_action = args.high_action
        self.fc1 = nn.Linear(args.obs_shape + args.action_shape, 256)
        # self.lstm2 = nn.LSTM(256, 128, 1)
        self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(512, 256)
        self.q_out = nn.Linear(128, 1)

    def forward(self, state, action):
        # print(state)
        # print(action)
        # state = torch.cat(state, dim=1)
        for i in range(len(action)):
            action[i] /= self.max_action
        # action = action.reshape(1280, 25)
        # print(state.shape)
        # print(action.shape)
        # action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)
        # x = torch.nn.init.xavier_uniform_(x, gain=1)
        # x = xavier_init(120, 128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value


import numpy as np
import torch
import os
from DDPG.ddpg import DDPG


class Agent:
    def __init__(self, args):
        self.args = args
        self.policy = DDPG(args)

    # action = agent.select_action(s[agent_id], self.noise, self.epsilon)
    def select_action(self, o, noise_rate): # o第i个智能体的观测
        #  归一化o

        # if np.random.uniform() < epsilon:   # epsilon = 0.1
        #     # 产生self.args.action_                       shape[self.agent_id]个随机数
        #     agent_actions = np.random.uniform(self.args.low_action, self.args.high_action, self.args.action_shape[self.agent_id])
        #     h = h_act_init
        #     c = c_act_init
        # else:
        # print(o)
        # o = np.array(o)
        # inputs = torch.from_numpy(o).float()
        inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
        # a.unsqueeze(0)增加维度（0表示，在第一个位置增加维度）1 * 60
        pi = self.policy.actor_network(inputs)  # 将观察送入actor网络并返回action
        pi = pi.squeeze(0)
        # h = h.squeeze(0)
        # c = c.squeeze(0)
        # print('{} : {}'.format(self.name, pi))
        # .cuda()是读取GPU中的数据 .data是读取Variable中的tensor .cpu是把数据转移到cpu上 .numpy()把tensor变成numpy
        # pi = pi.squeeze(0)
        agent_actions = pi.cpu().numpy()
        # print(agent_actions)
        noise = noise_rate * self.args.high_action * np.random.randn(*agent_actions.shape)  # gaussian noise
        # print(noise)
        # print('----------')
        agent_actions += noise
        agent_actions = np.clip(agent_actions, -self.args.high_action, self.args.high_action)  # 将值控制在-1到1之间
        return agent_actions.copy()

    def learn(self, transitions): #, h_cri_cur, c_cri_cur, h_cri_tar, c_cri_tar, h_act_tar, c_act_tar, h_act_cur, c_act_cur):
        # print(transitions)
        return self.policy.train(transitions)
        #, h_cri_cur, c_cri_cur, h_cri_tar, c_cri_tar, h_act_tar, c_act_tar, h_act_cur, c_act_cur)


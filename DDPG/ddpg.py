import torch
import os
import numpy as np
from DDPG.actor_critic import Actor, Critic
import torch.nn.functional as F

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else \
            np.zeros_like(self.mu)  # 目的是构建一个与mu同维度的数组，并初始化所有变量为零

class DDPG:
    def __init__(self, args):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        # self.agent_id = agent_id
        self.train_step = 0

        # create the network
        self.actor_network = Actor(args)
        self.critic_network = Critic(args)

        # build up the target network
        self.actor_target_network = Actor(args)
        self.critic_target_network = Critic(args)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        # , weight_decay=0.01

        self.loss_fun = torch.nn.MSELoss()

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent'
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # 加载模型
        if os.path.exists(self.model_path + '/actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
            self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
            print('Agent successfully loaded actor_network: {}'.format(self.model_path + '/actor_params.pkl'))
            print('Agent successfully loaded critic_network: {}'.format(self.model_path + '/critic_params.pkl'))

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # update the network
    def train(self, transitions): # , h_cri_cur, c_cri_cur, h_cri_tar, c_cri_tar, h_act_tar, c_act_tar, h_act_cur, c_act_cur):
        # torch.as_tensor(data, dtype=None,device=None)->Tensor : 为data生成tensor。
        # 如果data已经是tensor，且dtype和device与参数相同，则生成的tensor会和data共享内存。如果data是ndarray,
        # 且dtype对应，devices为cpu，则同样共享内存。其他情况则不共享内存。
        # 将transitions中的数转化为tensor
        # print(transitions)
        for key in transitions.keys():
            transitions[key] = torch.as_tensor(transitions[key], dtype=torch.float32)
        r = transitions['r']
        o = transitions['o']
        u = transitions['u']
        o_next = transitions['o_next']
        # print('奖励',r)
        # print('状态',o)
        # print('动作',u)
        # print('下一个状态',o_next)

        # calculate the target Q value function
        # u_next = [] # 下一时刻动作
        # h_tar= []
        # c_tar = []
        with torch.no_grad():
            # 得到下一个状态对应的动作
            a = self.actor_target_network(o_next)
            # print(a)
            # print(a)
            q_next = self.critic_target_network(o_next, a).detach()
            # q_next = q_next.detach()  # 切断一些分支的反向传播   返回一个新tensor，不计算梯度
            # print(q_next)
            # print('u_next', a)
            # print('q_next', q_next)
            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()
            # print('target_q', target_q)

        # the q loss
        q_value = self.critic_network(o, u)
        # print('q_value', q_value)
        # critic_loss = self.loss_fun(q_value, target_q)
        # critic_loss = (target_q - q_value).pow(2).mean()
        critic_loss = F.mse_loss(q_value, target_q)
        # print('critic_loss',critic_loss)
        self.critic_optim.zero_grad()
        # critic_loss.requires_grad_(True)
        critic_loss.backward()
        self.critic_optim.step()

        # self.critic_optim.zero_grad()
        # # critic_loss.requires_grad_(True)
        # critic_loss.backward()
        # self.critic_optim.step()

        # the actor loss
        # 重新选择联合动作中当前agent的动作，其他agent的动作不变
        u = self.actor_network(o)
        # print('u',u)
        actor_loss = - self.critic_network(o, u).mean()
        # print('actor_loss',actor_loss)
        # if self.agent_id == 0:
        #     print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
        # update the network
        self.actor_optim.zero_grad()
        # actor_loss.requires_grad_(True)
        actor_loss.backward()
        self.actor_optim.step()

        # self.critic_optim.zero_grad()
        # # critic_loss.requires_grad_(True)
        # critic_loss.backward()
        # self.critic_optim.step()

        self._soft_update_target_network()
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0: # save_rate: 2000
            self.save_model(self.train_step) # 每2000保存一下模型
        self.train_step += 1

        return actor_loss, critic_loss

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(),  model_path + '/' + num + '_critic_params.pkl')



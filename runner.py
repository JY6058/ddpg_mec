from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
from DDPG.ddpg import OUActionNoise
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class Runner:

    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        # self.epsilon = args.epsilon
        # self.noise = OUActionNoise(mu=np.zeros(args.action_shape))
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agent = Agent(args)
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    # def _init_agents(self):
    #     agents = []
    #     for i in range(self.args.num_agents):
    #         agent = Agent(i, self.args)
    #         agents.append(agent)
    #     return agents

    def run(self):
        SEED = 1
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        returns = []
        actor_loss11 = []
        critic_loss11 = []
        comp = []
        band = []
        hit_rate = []
        delay = []

        # noise_scale = 1.0
        # noise_decay = 0.9999
        # min_noise_scale = 0.1

        for time_step in tqdm(range(self.args.time_steps)):  # time_steps：100000
            # reset the environment  100次time step重置一次环境？
            if time_step % self.episode_limit == 0:  # 100
                s = self.env.reset()  # 初始化状态
                # self.noise.reset()

            # noise = noise_scale * self.noise()
            agent_actions = []  # 要放到buffer里，不需要映射到实际动作空间
            actions = []  # 将动作输出env.step，需要将动作映射到实际动作空间, 这里放在set_action执行
            # with torch.no_grad()或者@torch.no_grad()中的数据不需要计算梯度，也不会进行反向传播

            with torch.no_grad():
                # print(s)
                action = self.agent.select_action(s[0], self.noise)
                # print(action)
                agent_actions.append(action)
                actions.append(action)

            s_next, r, done, act, delay1, hit, cost = self.env.step(actions)
            # print(s_next)
            # print(act)

            # comp.append(act[0].trans_power[0])
            # band.append(act[0].trans_band[0])
            # hit_rate.append(hit[0])

            #  buffer 里的数据归一化
            # 将数据存入buffer

            # self.buffer.store_episode(s[:self.args.num_agents], agent_actions, r[:self.args.num_agents],
            #                           s_next[:self.args.num_agents], h_act_init, c_act_init, h_act, c_act, h_cri_init, c_cri_init)
            # s = s_next
            # h_act_init = h_act_cur
            # c_act_init = c_act_cur
            #  buffer 里的数据归一化
            # self.buffer.store_episode(s[:self.args.num_agents], agent_actions, r[:self.args.num_agents],
            #                           s_next[:self.args.num_agents])
            # s = s_next
            # h_act_init = h_act
            # c_act_init = c_act
            if self.buffer.current_size >= self.args.batch_size:  # batch-size = 256
                actor_loss1 = []
                critic_loss1 = []

                transitions = self.buffer.sample(self.args.batch_size)
                # print(transitions)
                # other_agents = self.agent.copy()
                # # remove() 函数用于移除列表中某个值的第一个匹配项。移除当前agent
                # other_agents.remove(agent)

                actor_loss, critic_loss = self.agent.learn(transitions)
                actor_loss1.append(actor_loss.item())
                critic_loss1.append(critic_loss.item())
                # print(actor_loss.item())
                # print(critic_loss.item())

                actor_loss11.append(actor_loss1[0])
                critic_loss11.append(critic_loss1[0])

                self.buffer.store_episode(s, agent_actions, r, s_next)
            else:
                self.buffer.store_episode(s, agent_actions, r, s_next)
                # print(s)
            s = s_next
            # h_act_init = h_act
            # c_act_init = c_act

            # actor loss图像
            plt.figure("Actor Loss")
            plt.plot(range(len(actor_loss11)), actor_loss11)
            plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
            plt.ylabel('actor loss')
            plt.savefig(self.save_path + '/actor_loss.pdf', bbox_inches="tight", format='pdf')

            # critic loss图像
            plt.figure("Critic Loss")
            plt.plot(range(len(critic_loss11)), critic_loss11)
            plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
            plt.ylabel('critic_loss')
            plt.savefig(self.save_path + '/critic_loss.pdf', bbox_inches="tight", format='pdf')

            plt.close('all')

            if time_step > 0 and time_step % self.args.evaluate_rate == 0:  # evaluate_rate：1000
                rew, computing, bandwidth, delay1, hit = self.evaluate()

                returns.append(rew)
                comp.append(computing)
                band.append(bandwidth)
                hit_rate.append(hit)
                delay.append(delay1)

                plt.figure("reward")
                plt.plot(range(len(returns)), returns)
                plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
                plt.ylabel('average returns')
                plt.savefig(self.save_path + '/reward.pdf', bbox_inches="tight", format='pdf')

                plt.figure("delay")
                plt.plot(range(len(delay)), delay)
                plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
                plt.ylabel('average delay')
                plt.savefig(self.save_path + '/delay.pdf', bbox_inches="tight", format='pdf')

                # 计算资源分配图像
                plt.figure("Computing Resources")
                plt.plot(range(len(comp)), comp)
                plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
                plt.ylabel('Computing Resources')
                plt.savefig(self.save_path + '/computing_resources.pdf', bbox_inches="tight", format='pdf')

                # 频谱资源分配图像
                plt.figure("Spectrum Resource")
                plt.plot(range(len(band)), band)
                plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
                plt.ylabel('Spectrum Resource')
                plt.savefig(self.save_path + '/spectrum_resource.pdf', bbox_inches="tight", format='pdf')

                # 缓存命中率图像
                plt.figure("hit rate")
                plt.plot(range(len(hit_rate)), hit_rate)
                plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
                plt.ylabel('hit rate')
                plt.savefig(self.save_path + '/hit_rate.pdf', bbox_inches="tight", format='pdf')

            plt.close('all')

            self.noise = max(0.05, self.noise - 0.0000005)
            # self.noise = max(0.05, self.noise - 0.000001)
            # self.noise = max(0.05, self.noise - 0.000005)
            # self.noise = max(0.05, 0.999 * self.noise)
            # self.epsilon = max(0.05, self.epsilon - 0.0000005)
            # noise_scale = max(noise_scale * noise_decay, min_noise_scale)

            # plt.close('all')

            np.save(self.save_path + '/reward.pkl', returns)
            np.save(self.save_path + '/actor_loss.pkl', actor_loss11)
            np.save(self.save_path + '/critic_loss.pkl', critic_loss11)
            np.save(self.save_path + '/band_allocation.pkl', band)
            np.save(self.save_path + '/comp_allocation.pkl', comp)
            np.save(self.save_path + '/hit_rite.pkl', hit_rate)
            np.save(self.save_path + '/delay.pkl', delay)

        file_path = self.save_path + '/critic_loss.txt'
        self.save_to_txt(critic_loss11, file_path)
        file_path = self.save_path + '/actor_loss.txt'
        self.save_to_txt(actor_loss11, file_path)
        file_path = self.save_path + '/reward.txt'
        self.save_to_txt(returns, file_path)
        file_path = self.save_path + '/band_allocation.txt'
        self.save_to_txt(band, file_path)
        file_path = self.save_path + '/comp_allocation.txt'
        self.save_to_txt(comp, file_path)
        file_path = self.save_path + '/hit_rate.txt'
        self.save_to_txt(hit_rate, file_path)
        file_path = self.save_path + '/delay.txt'
        self.save_to_txt(delay, file_path)

    def evaluate(self):
        returns = []
        comp = []
        band = []
        hit_rate = []
        delay = []
        for episode in range(self.args.evaluate_episodes):  # evaluate_episodes：5
            # reset the environment
            s = self.env.reset()
            rewards = 0
            comp1 = []
            band1 = []
            comp11 = 0
            band11 = 0
            hit_rate1 = 0
            delay1 = 0

            for time_step in range(self.args.evaluate_episode_len):  # evaluate_episode_len：100
                self.env.render()
                actions = []
                with torch.no_grad():
                    action = self.agent.select_action(s[0], 0)
                    # print(action)
                    actions.append(action)

                # print(actions)
                # actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])

                s_next, r, done, act, time_delay, hit, cost = self.env.step(actions)
                # print(r)
                # print("hit的值")
                # print(hit)
                # print(act[0].trans_power[0])
                comp1.append(act[0].trans_power[0])
                band1.append(act[0].trans_band[0])
                hit_rate1 += sum(hit) / self.args.num_UEs
                # hit_local_rate1 += sum(hit_local_bs) / (self.args.num_agents * self.args.num_UEs)
                # hit_bs1.append(hit[0] / self.args.num_UEs)

                # s = np.array(s[0])
                # requested_service = np.round((((s[:15] - 0) * (10 - 1)) / 1) + 0)
                # task_size = np.round((((s[15:30] - 0) * 20) / 1) + 0) + 10  # 10-30
                # # print(agent.state.delay_tolerance)
                # delay_tolerance = np.round((((s[30:45] - 0) * 20) / 1) + 0) + 10
                # # print(delay_tolerance)
                # print(s)
                # print(requested_service)
                # print(task_size)
                # print(delay_tolerance)

                rewards += r  # 100个时隙下一个基站下的所有用户的总时延
                # print(r) 值
                # print(act[0].offloading)
                # print(act[0].caching)
                delay1 += sum(time_delay)
                # print(time_delay) 是一个列表，里有一个值
                s = s_next

            print(act[0].offloading)
            print(act[0].association)
            # # print(act[0].trans_band)
            # # print(act[0].trans_power)
            print(cost)

            rewards = rewards / self.args.evaluate_episode_len
            comp11 = sum(comp1) / self.args.evaluate_episode_len
            band11 = sum(band1) / self.args.evaluate_episode_len
            delay1 = delay1 / self.args.evaluate_episode_len
            hit_rate1 = hit_rate1 / self.args.evaluate_episode_len

            returns.append(rewards)
            comp.append(comp11)
            band.append(band11)
            delay.append(delay1)
            hit_rate.append(hit_rate1)
            # returns = returns / 100
            # 100个time_step打印一次10个episode的奖励
            # 平均每个time_step下一个基站下的所有用户的时延
            # 100个time_step为1个episode
            # print('Returns is', rewards / 100)
            print("reward: {:02.4f}, comp: {:02.4f}, band: {:02.4f}, delay: {:02.4f}, hit_rate: {:02.4f}".format(rewards, comp11, band11, delay1, hit_rate1))


        return sum(returns) / self.args.evaluate_episodes, sum(comp) / self.args.evaluate_episodes, \
               sum(band) / self.args.evaluate_episodes,  sum(delay) / self.args.evaluate_episodes, sum(hit_rate) / self.args.evaluate_episodes

    def save_to_txt(self, curr_reward, file_path):
        with open(file_path, 'w') as f:
            for i in range(len(curr_reward)):
                f.write("%s \n" % curr_reward[i])

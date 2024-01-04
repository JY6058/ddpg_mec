import threading
import numpy as np


class Buffer:
    def __init__(self, args):
        self.size = args.buffer_size  # 经验池大小
        self.args = args
        self.current_size = 0
        self.buffer = dict()

        self.buffer['o'] = np.empty([self.size, self.args.obs_shape])
        self.buffer['u'] = np.empty([self.size, self.args.action_shape])
        self.buffer['r'] = np.empty([self.size])
        self.buffer['o_next'] = np.empty([self.size, self.args.obs_shape])
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, o, u, r, o_next):
        idxs = self._get_storage_idx(inc=1)
        # 以transition的形式存，每次只存一条经验
        # print(o)
        # print(u)
        # print(r)
        # print(o_next)
        # o = np.array(o)
        # u = np.array(u)
        # r = np.array(r)
        # o_next = np.array(o_next)
        with self.lock:
            self.buffer['o'][idxs] = o[0]
            self.buffer['u'][idxs] = u[0]
            self.buffer['r'][idxs] = r
            self.buffer['o_next'][idxs] = o_next[0]

    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idx]
        return temp_buffer

    # 获取存储位置的函数
    def _get_storage_idx(self, inc=None):
        inc = inc or 1  # 1
        # print("inc的值")
        # print(inc)
        if self.current_size + inc <= self.size: # 5e5
            idx = np.arange(self.current_size, self.current_size + inc)  # 起点， 终点
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            # np.random.randint(low, high=None, size=None, dtype='l') 用于生成指定范围内的随机整数
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])   # 串联数组
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx



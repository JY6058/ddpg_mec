# An old version of OpenAI Gym's multi_discrete.py. (Was getting affected by Gym updates)
# (https://github.com/openai/gym/blob/1fb81d4e3fb780ccf77fec731287ba07da35eb84/gym/spaces/multi_discrete.py)

import numpy as np

import gym
from gym.spaces import prng

class MultiDiscrete(gym.Space):
    """
    - The multi-discrete action space consists of a series of discrete action spaces with different parameters
    - It can be adapted to both a Discrete action space or a continuous (Box) action space
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of arrays containing [min, max] for each discrete action space
       where the discrete action space can take any integers from `min` to `max` (both inclusive)
    Note: A value of 0 always need to represent the NOOP action.
    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:
        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    - Can be initialized as
        MultiDiscrete([ [0,4], [0,1], [0,1] ])
    - 多离散动作空间由一系列具有不同参数的离散动作空间组成
    - 它可以适应离散动作空间或连续（盒子）动作空间
    - 表示游戏控制器或键盘很有用，其中每个键都可以表示为离散的动作空间
    - 通过为每个离散的动作空间传递一个包含 [min, max] 的数组来对其进行参数化
       其中离散动作空间可以采用从 `min` 到 `max` 的任何整数（包括两者）
    注意：值 0 始终需要表示 NOOP 操作。
    例如任天堂游戏控制器
    - 可以概念化为 3 个离散的动作空间：
        1) 箭头键：离散 5 - NOOP[0]、UP[1]、RIGHT[2]、DOWN[3]、LEFT[4] - 参数：最小值：0，最大值：4
        2) 按钮 A：离散 2 - NOOP[0]，Pressed[1] - 参数：最小值：0，最大值：1
        3) 按钮 B：离散 2 - NOOP[0]，Pressed[1] - 参数：最小值：0，最大值：1
    - 可以初始化为
        多离散([ [0,4], [0,1], [0,1] ])
    """
    def __init__(self, array_of_param_array):
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]

    def sample(self):
        """ Returns a array with one sample from each discrete action space """
        # 返回一个数组，其中包含来自每个离散动作空间的一个样本
        # For each row: round(random .* (max - min) + min, 0)
        random_array = prng.np_random.rand(self.num_discrete_space)
        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.), random_array) + self.low)]

    def contains(self, x):
        return len(x) == self.num_discrete_space and (np.array(x) >= self.low).all() and (np.array(x) <= self.high).all()

    @property
    def shape(self):
        return self.num_discrete_space

    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)

    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)

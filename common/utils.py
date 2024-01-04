import numpy as np
import inspect
import functools
# from myUnits import randomwalk


# service_ue1, service_ue2, service_ue3, service_ue4, service_ue5 = randomwalk()
def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def make_env(args):
    from singleagent.environment import AgentEnv
    import singleagent.scenarios as scenarios

    # service_ue1, service_ue2, service_ue3, service_ue4, service_ue5 = randomwalk()

    # load scenario from script
    scenario = scenarios.load(args.scenario_name + ".py").Scenario()

    # create world  各种实体？
    world = scenario.make_world()
    # create multiagent environment
    env = AgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)   # , scenario.battery)
    # env = MultiAgentEnv(world)
    # args.num_agents = env.n  # 智能体的数量5
    # args.num_channels = env.num_channels  # 信道的数量
    args.num_service = env.max_service_type  # 服务类型的数量 30
    obs_shape = []
    # 获得每个agent观测空间的大小，60
    # print(env.observation_space)
    for content in env.observation_space:
        obs_shape.append(content.n)
        # print(content.n)
        # print("obs_shape的大小")
        # print(obs_shape)
    args.obs_shape = obs_shape[0]
    # print("args.obs_shape的大小")
    # print(args.obs_shape)
    # shape为动作空间的大小 for each agent
    action_space = []
    size = 0
    # # 获得每个智能体动作空间的大小60
    for j in range(len(env.action_space[0].spaces)):
        if j == 0:
            size += env.action_space[0].spaces[0].shape[0]
        else:
            size += env.action_space[0].spaces[j].shape[0] * env.action_space[0].spaces[j].shape[1]
    action_space.append(size)
    args.action_shape = action_space[0]  # 每一维代表该agent的act维度
    # print(args.action_shape)

    args.high_action = 1
    args.low_action = -1
    args.num_UEs = env.agent.num_UEs
    # args.high_offloading = args.num_agents
    # args.offloading_low_action = 0
    # args.channel_high_action = args.num_channels
    # args.channel_low_action = 0
    # args.caching_high_action = args.num_service
    # args.caching_low_action = 1
    # args.action_shape = [sum(env.action_space[i].high - env.action_space[i].low + 1) for i in range(args.num_agents)]
    return env, args

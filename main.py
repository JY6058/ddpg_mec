from runner import Runner
from common.arguments import get_args
from common.utils import make_env

import numpy as np
import random
import torch


if __name__ == '__main__':
    # get the params
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    env, args = make_env(args)
    seed = 1
    env.seed(seed)
    np.random.seed(seed)
    # h_act_init = torch.zeros(5, 1, 128)
    # c_act_init = torch.zeros(5, 1, 128)

    runner = Runner(args, env)
    if args.evaluate:
        returns = runner.evaluate()
        print('Average returns is', returns)
    else:
        runner.run()

# -----------------------------------------------------------------------------------------------
# © 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research Institute" (AIRI);
# Moscow Institute of Physics and Technology (National Research University). All rights reserved.
# 
# Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
# -----------------------------------------------------------------------------------------------

import numpy as np

from htm_rl.agents.agent import Agent
from htm_rl.common.sdr import SparseSdr
from htm_rl.envs.env import Env


class RndAgent(Agent):
    n_actions: int

    def __init__(
            self,
            env: Env,
            seed: int,
    ):
        self.n_actions = env.n_actions
        self.rng = np.random.default_rng(seed)

    @property
    def name(self):
        return 'rnd'

    def act(self, reward: float, state: SparseSdr, first: bool):
        return self.rng.integers(self.n_actions)

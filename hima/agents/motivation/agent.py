#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
from hima.modules.motivation import Amygdala, Striatum
from hima.common.sdr import SparseSdr
from hima.common.config_utils import TConfig


class Agent:
    def __init__(self, obs_dim: int, action_dim: int, config: TConfig):
        self.seed = config['seed']
        self._rng = np.random.default_rng(self.seed)
        self.action_dim = action_dim

        self.amg = Amygdala(
            sdr_size=obs_dim,
            seed=self.seed, **config['amygdala']
        )

        self.str = Striatum(
            input_size=obs_dim, seed=self.seed, output_size=action_dim,
            **config['striatum']
        )

    def act(self, obs: SparseSdr, reward: float, is_first: bool) -> int:
        if is_first:
            self.amg.reset()
            self.str.reset()

        amg_sdr = self.amg.compute(obs)
        self.amg.update(obs, reward)
        action = self.str.compute(obs, reward, True)
        return action

    def get_amg_value(self, obs: SparseSdr) -> float:
        return self.amg.get_value(obs)

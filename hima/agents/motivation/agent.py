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
            input_size=self.amg.sdr_size+obs_dim, seed=self.seed, output_size=action_dim,
            **config['striatum']
        )

    def act(self, obs: SparseSdr, reward: float, is_first: bool) -> int:
        sdr_input = np.concatenate(
            (
                self.amg.compute(obs), self.amg.sdr_size + obs
            )
        )
        actions = self.str.compute(sdr_input, 1, True)

        if is_first:
            self.amg.reset()
            self.str.reset()

        self.amg.update(obs, reward)
        actions = actions / actions.sum()
        action = self._rng.choice(self.action_dim, 1, p=actions)[0]
        return action

    def get_amg_value(self, obs: SparseSdr) -> float:
        return self.amg.get_value(obs)

    def get_q_values(self, obs: SparseSdr) -> np.ndarray:
        sdr_input = np.concatenate(
            (
                self.amg.compute(obs), self.amg.sdr_size + obs
            )
        )
        values = self.str.compute(sdr_input, 1, False)
        return values

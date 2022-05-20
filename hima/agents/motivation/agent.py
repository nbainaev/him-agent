#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
from hima.modules.motivation import Amygdala, StriatumBlock, Policy
from hima.common.sdr import SparseSdr
from hima.common.config_utils import TConfig


class Agent:
    def __init__(self, obs_dim: int, action_dim: int, config: TConfig):
        self.seed = config['seed']
        self._rng = np.random.default_rng(self.seed)

        self.amg = Amygdala(
            sdr_size=obs_dim,
            seed=self.seed, **config['amygdala']
        )

        self.amg_striatum = StriatumBlock(
            inputDimensions=[1, self.amg.sdr_size],
            seed=self.seed, **config['striatum']
        )
        self.sma_striatum = StriatumBlock(
            inputDimensions=[1, obs_dim],
            seed=self.seed, **config['striatum']
        )
        self.striatum_output_sdr_size = self.amg_striatum.output_sdr_size \
                                        + self.sma_striatum.output_sdr_size

        self.policy = Policy(
            sdr_size=self.striatum_output_sdr_size, seed=self.seed,
            n_actions=action_dim, **config['policy']
        )

    def act(self, obs: SparseSdr, reward: float, is_first: bool) -> int:
        sdr_amg = self.amg_striatum.compute(self.amg.compute(obs), 0, True)
        sdr_sma = self.sma_striatum.compute(obs, 0, True)
        sdr_str = np.concatenate(
            (
                sdr_amg, self.amg_striatum.output_sdr_size + sdr_sma
            )
        )

        if is_first:
            self.amg.reset()
            self.sma_striatum.reset()
            self.amg_striatum.reset()
            self.policy.reset()

        self.amg.update(obs, reward)
        action = self.policy.compute(sdr_str)
        sa = action * self.policy.state_size + sdr_str
        self.policy.update(sa, reward)
        return action

    def get_amg_value(self, obs: SparseSdr) -> float:
        return self.amg.get_value(obs)

    def get_q_values(self, obs: SparseSdr) -> np.ndarray:
        sdr_amg = self.amg_striatum.compute(self.amg.compute(obs), 0, False)
        sdr_sma = self.sma_striatum.compute(obs, 0, False)
        sdr_str = np.concatenate(
            (
                sdr_amg, self.amg_striatum.output_sdr_size + sdr_sma
            )
        )
        return self.policy.get_values(sdr_str)

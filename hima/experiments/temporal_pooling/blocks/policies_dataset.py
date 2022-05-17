#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from typing import Iterator

import numpy as np
from numpy.random import Generator

from hima.common.sdr import SparseSdr
from hima.common.sds import Sds


class Policy:
    id: int
    _policy: np.ndarray

    def __init__(self, id_: int, policy, seed=None):
        self.id = id_
        self._policy = policy

    def __iter__(self) -> Iterator[tuple[SparseSdr, SparseSdr]]:
        return iter(self._policy)

    def shuffle(self) -> None:
        ...


class SyntheticDatasetBlock:
    name: str
    n_states: int
    n_actions: int

    context_sds: Sds
    output_sds: Sds

    _policies: list[Policy]
    _rng: Generator

    def __init__(
            self, n_states: int, states_sds: Sds, n_actions: int, actions_sds: Sds,
            policies: list[Policy], seed: int
    ):
        self.n_states = n_states
        self.context_sds = states_sds
        self.n_actions = n_actions
        self.output_sds = actions_sds
        self._policies = policies

        self._rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[Policy]:
        return iter(self._policies)

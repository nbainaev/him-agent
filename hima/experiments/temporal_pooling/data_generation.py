#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from typing import Optional, Iterator

import numpy as np
from numpy.random import Generator


class SequenceSelector:
    n_elements: int
    regime: str
    seed: Optional[int]

    indices: np.ndarray

    def __init__(self, n_elements: int, regime: str, seed: int = None):
        self.n_elements = n_elements
        self.regime = regime
        self.seed = seed
        self.indices = self._select_order()

    def reshuffle(self):
        self.seed = np.random.default_rng(self.seed).integers(100000)
        self.indices = self._select_order()

    def __iter__(self) -> Iterator[int]:
        if self.regime == 'unordered':
            # re-shuffle each time making the order to be random
            self.reshuffle()

        return iter(self.indices)

    def _select_order(self) -> np.ndarray:
        if self.regime == 'ordered':
            return np.arange(self.n_elements)
        elif self.regime == 'shuffled' or self.regime == 'unordered':
            rng = np.random.default_rng(self.seed)
            return rng.permutation(self.n_elements)
        else:
            raise KeyError(f'{self.regime} is not supported')


def generate_data(n, n_actions, n_states, randomness=1.0, seed=0):
    raw_data = list()
    np.random.seed(seed)
    seed_seq = np.random.randint(0, n_actions, n_states)
    raw_data.append(seed_seq.copy())
    n_replace = int(n_states * randomness)
    for i in range(1, n):
        new_seq = np.random.randint(0, n_actions, n_states)
        if randomness == 1.0:
            raw_data.append(new_seq)
        else:
            indices = np.random.randint(0, n_states, n_replace)
            seed_seq[indices] = new_seq[indices]
            raw_data.append(seed_seq.copy())
    data = [list(zip(range(n_states), x)) for x in raw_data]
    return raw_data, data

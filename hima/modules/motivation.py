#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
from hima.common.sdr import SparseSdr
from hima.common.utils import update_exp_trace


class ValueNetwork:
    def __init__(
            self, seed: int, sdr_size: int, discount_factor: float, learning_rate: float,
    ):
        self._rng = np.random.default_rng(seed)

        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        self.cell_value = self._rng.uniform(-1e-5, 1e-5, size=sdr_size)

    def update(
            self, sdr: SparseSdr, reward: float, next_sdr: SparseSdr, e_traces: np.ndarray
    ):
        lr = self.learning_rate
        v = self.cell_value
        td_err = self.td_error(sdr, reward, next_sdr)
        v += lr * td_err * e_traces

    def td_error(self, sdr: SparseSdr, reward: float, next_sdr: SparseSdr):
        gamma = self.discount_factor
        r = reward
        v_sdr = self.value(sdr)
        next_v_sdr = self.value(next_sdr)
        return r + gamma * next_v_sdr - v_sdr

    def value(self, x: SparseSdr) -> float:
        if len(x) == 0:
            return 0
        return np.median(self.cell_value[x])


class EligibilityTraces:
    def __init__(
            self, sdr_size: int, trace_decay: float,
            discount_factor: float, with_reset: bool = False
    ):
        self.sdr_size = sdr_size
        self.trace_decay = trace_decay
        self.discount_factor = discount_factor
        self.with_reset = with_reset
        self.cell_traces = None
        self.reset()

    def update(self, sdr: SparseSdr):
        lambda_, gamma = self.trace_decay, self.discount_factor
        update_exp_trace(
            self.cell_traces, sdr,
            decay=lambda_ * gamma,
            with_reset=self.with_reset
        )

    def reset(self):
        if self.cell_traces is None:
            self.cell_traces = np.zeros(self.sdr_size, dtype=np.float)
        self.cell_traces.fill(0.)


class Unit2DEncoder:
    def __init__(self, n_buckets: int, bucket_shape: tuple[int, int]):
        self.n_buckets = n_buckets
        self.bucket_shape = bucket_shape
        self.sdr_shape = (bucket_shape[0], n_buckets * bucket_shape[1])
        self.sdr_size = self.sdr_shape[0] * self.sdr_shape[1]

        self.base_sdr = np.array([
            self.sdr_shape[1] * i + np.arange(bucket_shape[1]) for i in range(bucket_shape[0])
        ], dtype=int).flatten()

    def compute(self, values: list[float]):
        out = np.empty(self.base_sdr.size * len(values), dtype=int)
        for ind, value in enumerate(values):
            if value < 0:
                x = 0
            elif value > 1:
                x = 1
            else:
                x = value
            start = int(x * (self.sdr_shape[1] - self.bucket_shape[1]))
            start += ind * self.sdr_size
            out[ind * self.base_sdr.size: (ind + 1) * self.base_sdr.size] = start + self.base_sdr
        return out


class Amygdala:
    def __init__(
            self, seed: int, sdr_size: int, gamma: float,
            alpha: float, lambda_: float, with_reset: bool,
            bucket_shape: tuple[int, int], n_buckets: int,
            min_cut_fraction: float = 0.05
    ):
        self.sdr_size = sdr_size
        self.min_cut_fraction = min_cut_fraction
        self.encoder = Unit2DEncoder(n_buckets, bucket_shape)
        self.out_sdr_shape = (2 * self.encoder.sdr_shape[0], self.encoder.sdr_shape[1])
        self.out_sdr_size = 2 * self.encoder.sdr_size

        self.gamma = gamma
        self.alpha = alpha
        self.value_network = ValueNetwork(seed, sdr_size, gamma, alpha)
        self.eligibility_traces = EligibilityTraces(sdr_size, lambda_, gamma, with_reset)
        self.current_sdr = None
        self.current_reward = None

    def compute(self, sdr: SparseSdr, dopamine: float) -> SparseSdr:
        min_ = np.quantile(self.value_network.cell_value, self.min_cut_fraction)
        max_ = np.max(self.value_network.cell_value)
        value = (self.value_network.value(sdr) - min_) / (max_ - min_)
        if value <= 0:
            value = 0
        d1 = np.tanh(value / (1 + dopamine))
        d2 = 1 - np.tanh((1 + dopamine) * value)
        output = self.encoder.compute([d1, d2])
        return output

    def update(self, sdr: np.ndarray, reward: float):
        if self.current_sdr is None:
            self.current_sdr = np.copy(sdr)
            self.current_reward = reward
            self.eligibility_traces.update(sdr)
            return

        prev_sdr, prev_rew = self.current_sdr, self.current_reward
        self.value_network.update(prev_sdr, prev_rew, sdr, self.eligibility_traces.cell_traces)
        self.eligibility_traces.update(sdr)
        self.current_reward = reward
        self.current_sdr = np.copy(sdr)

    def reset(self):
        if self.current_sdr is not None:
            prev_sdr, prev_rew = self.current_sdr, self.current_reward
            self.value_network.update(prev_sdr, prev_rew, [], self.eligibility_traces.cell_traces)
            self.current_sdr, self.current_reward = None, None
            self.eligibility_traces.reset()

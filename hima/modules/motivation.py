#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
from hima.common.sdr import SparseSdr
from hima.common.utils import update_exp_trace
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.sdr import SDR


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

    def compute(self, value: float):
        out = np.empty(self.base_sdr.size, dtype=int)
        if value < 0:
            x = 0
        elif value > 1:
            x = 1
        else:
            x = value
        start = int(x * (self.sdr_shape[1] - self.bucket_shape[1]))
        out[:] = start + self.base_sdr
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
        self.out_sdr_shape = self.encoder.sdr_shape
        self.out_sdr_size = self.encoder.sdr_size

        self.gamma = gamma
        self.alpha = alpha
        self.value_network = ValueNetwork(seed, sdr_size, gamma, alpha)
        self.eligibility_traces = EligibilityTraces(sdr_size, lambda_, gamma, with_reset)
        self.current_sdr = None
        self.current_reward = None

    def compute(self, sdr: SparseSdr) -> SparseSdr:
        value = self.get_value(sdr)
        output = self.encoder.compute(value)
        return output

    def get_value(self, sdr: SparseSdr) -> float:
        min_ = np.quantile(self.value_network.cell_value, self.min_cut_fraction)
        max_ = np.max(self.value_network.cell_value)
        value = (self.value_network.value(sdr) - min_) / (max_ - min_)
        if value < 0:
            value = 0
        return value

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


class StriatumBlock:
    def __init__(
            self, inputDimensions: list[int], columnDimensions: list[int], potentialRadius: int,
            potentialPct: float, globalInhibition: bool, localAreaDensity: float,
            stimulusThreshold: int, synPermInactiveDec: float, synPermActiveInc: float,
            synPermConnected: float, minPctOverlapDutyCycle: float, dutyCyclePeriod: int,
            boostStrength: float, seed: int, wrapAround: bool, dopamine_factor: float
    ):
        self.output_sdr_shape = (2 * columnDimensions[0], columnDimensions[1])
        self.zone_size = columnDimensions[0]
        self.sp = SpatialPooler(
                inputDimensions=inputDimensions,
                columnDimensions=self.output_sdr_shape,
                potentialPct=potentialPct,
                potentialRadius=potentialRadius,
                globalInhibition=globalInhibition,
                localAreaDensity=localAreaDensity,
                stimulusThreshold=stimulusThreshold,
                synPermInactiveDec=synPermInactiveDec,
                synPermActiveInc=synPermActiveInc,
                synPermConnected=synPermConnected,
                minPctOverlapDutyCycle=minPctOverlapDutyCycle,
                dutyCyclePeriod=dutyCyclePeriod,
                boostStrength=boostStrength,
                seed=seed,
                wrapAround=wrapAround
        )

        self.output_sdr_size = self.sp.getNumColumns()
        self._input_sdr = SDR(inputDimensions)
        self._output_sdr = SDR(self.output_sdr_shape)
        self._boost_factors = np.ones(self.output_sdr_shape, dtype=np.float32)

        self.dopamine_level = 0
        self.dopamine_factor = dopamine_factor

    def update_dopamine_boost(self, dopamine: float):
        self.dopamine_level *= self.dopamine_factor
        self.dopamine_level += dopamine

        self.sp.getBoostFactors(self._boost_factors)
        self._boost_factors[:self.zone_size, :] *= (1 + self.dopamine_level)
        self._boost_factors[self.zone_size:, :] /= (1 + self.dopamine_level)
        self.sp.setBoostFactors(self._boost_factors)

    def compute(self, sdr: SparseSdr, dopamine: float, learn: bool) -> SparseSdr:
        if learn:
            self.update_dopamine_boost(dopamine)

        self._input_sdr.sparse = sdr
        self.sp.compute(self._input_sdr, learn=learn, output=self._output_sdr)
        output = np.copy(self._output_sdr.sparse)
        return output

    def reset(self):
        self.dopamine_level = 0

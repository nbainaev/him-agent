#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
from hima.common.sdr import SparseSdr
from hima.common.utils import update_exp_trace
from htm.bindings.algorithms import Connections
from htm.bindings.sdr import SDR
from hima.common.utils import softmax


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


class TDLambda:
    def __init__(
            self, seed: int, sdr_size: int, gamma: float,
            alpha: float, lambda_: float, with_reset: bool,
    ):
        self.sdr_size = sdr_size

        self.gamma = gamma
        self.alpha = alpha
        self.value_network = ValueNetwork(seed, sdr_size, gamma, alpha)
        self.eligibility_traces = EligibilityTraces(sdr_size, lambda_, gamma, with_reset)
        self.current_sdr = None
        self.current_reward = None

    def get_value(self, sdr: SparseSdr) -> float:
        value = self.value_network.value(sdr)
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


class Amygdala(TDLambda):
    def __init__(
            self, seed: int, sdr_size: int, gamma: float,
            alpha: float, lambda_: float, with_reset: bool,
            filter_factor: float = 0.2
    ):
        super().__init__(seed, sdr_size, gamma, alpha, lambda_, with_reset)
        self.filter_factor = filter_factor

    def compute(self, sdr: SparseSdr) -> SparseSdr:
        mask = self.get_mask()
        output = np.intersect1d(sdr, mask)
        return output

    def get_mask(self) -> SparseSdr:
        values = self.value_network.cell_value
        threshold = np.quantile(values, 1 - self.filter_factor)
        mask = np.flatnonzero(values > threshold)
        return mask

    def get_masked_values(self):
        output = np.zeros_like(self.value_network.cell_value)
        mask = self.get_mask()
        output[mask] = self.value_network.cell_value[mask]
        return output

    def get_value(self, sdr: SparseSdr) -> float:
        min_ = np.min(self.value_network.cell_value)
        max_ = np.max(self.value_network.cell_value)
        value = (self.value_network.value(sdr) - min_) / (max_ - min_)
        return value


class Policy(TDLambda):
    def __init__(
            self, seed: int, sdr_size: int, gamma: float,
            alpha: float, lambda_: float, with_reset: bool,
            n_actions: int, temperature: float
    ):
        super().__init__(seed, sdr_size * n_actions, gamma, alpha, lambda_, with_reset)
        self.n_actions = n_actions
        self.state_size = sdr_size
        self.temperature = temperature
        self._rng = np.random.default_rng(seed)

    def get_values(self, state: SparseSdr) -> np.ndarray:
        values = np.array([
            self.get_value(a * self.state_size + state) for a in range(self.n_actions)
        ])
        return softmax(values, self.temperature)

    def compute(self, state: SparseSdr) -> int:
        values = self.get_values(state)
        action = self._rng.choice(self.n_actions, p=values)
        return action


class Striatum:
    def __init__(
            self, input_size: int, output_size: int, field_size: int, synapse_threshold: float,
            potential_pct: float, seed: int, connected_pct: float, d1_pct: float, beta: float,
            stimulus_threshold: float, active_neurons: int, syn_increment: float,
            syn_decrement: float
    ):
        self.input_size = input_size
        self.spn_size = output_size * field_size
        self.output_size = output_size

        self.stimulus_threshold = stimulus_threshold
        self.n_active_neurons = active_neurons
        self.syn_increment = syn_increment
        self.syn_decrement = syn_decrement
        self.beta = beta
        self._rng = np.random.default_rng(seed)

        # connections from input to intermediate neurons
        self.connections = Connections(self.spn_size, synapse_threshold)
        n_potential_cells = int(potential_pct * input_size)
        for cell in range(self.spn_size):
            self.connections.createSegment(cell, 1)
            potential_cells = self._rng.permutation(input_size)[:n_potential_cells]
            for presyn_cell in potential_cells:
                if self._rng.uniform() < connected_pct:
                    permanence = self._rng.uniform(synapse_threshold, 1)
                else:
                    permanence = self._rng.uniform(0, synapse_threshold)
                self.connections.createSynapse(cell, presyn_cell, permanence)

        # connections from intermediate to output neurons
        n_d1_cells = int(d1_pct * field_size)
        n_d2_cells = field_size - n_d1_cells
        self.d1_mask = np.empty((output_size * n_d1_cells), dtype=int)
        self.d2_mask = np.empty((output_size * n_d2_cells), dtype=int)
        for cell in range(output_size):
            inds = self._rng.permutation(field_size)
            d1 = inds[:n_d1_cells]
            d2 = inds[n_d1_cells:]
            self.d1_mask[cell * n_d1_cells: (cell + 1) * n_d1_cells] = d1 + field_size * cell
            self.d2_mask[cell * n_d2_cells: (cell + 1) * n_d2_cells] = d2 + field_size * cell

        self.dopa_factors = np.ones(self.spn_size)
        self.boost_factors = np.ones(self.spn_size)
        self._input_sdr = SDR(input_size)
        self._spn = np.zeros(self.spn_size)

    def update_dopamine_boost(self, dopamine: float):
        pass

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def compute(self, sdr: SparseSdr, dopamine: float, learn: bool) -> np.ndarray:
        self._input_sdr.sparse = sdr
        activity = self.connections.computeActivity(self._input_sdr, learn)

        adb = activity * self.dopa_factors * self.boost_factors
        theta = self.stimulus_threshold
        d = dopamine

        self._spn[self.d1_mask] = adb[self.d1_mask] - theta / d
        self._spn[self.d2_mask] = adb[self.d2_mask] - theta * d
        self._spn[self._spn < 0] = 0

        k = self.spn_size - self.n_active_neurons
        spn_out = np.argpartition(self._spn, k)[k:]
        possible_output = np.nonzero(self._spn)
        spn_out = np.intersect1d(spn_out, possible_output)

        if learn:
            for cell in spn_out:
                self.connections.adaptSegment(
                    cell, self._input_sdr, self.syn_increment, self.syn_decrement
                )
                self.connections.raisePermanencesToThreshold(cell, self.stimulus_threshold)

        output = np.zeros(self.output_size)
        for cell in range(self.output_size):
            output_d1 = np.intersect1d(spn_out, self.d1_mask.reshape((self.output_size, -1))[cell])
            output_d2 = np.intersect1d(spn_out, self.d2_mask.reshape((self.output_size, -1))[cell])
            output[cell] = self.sigmoid(self.beta * (len(output_d1) - len(output_d2)))
        return output

    def reset(self):
        pass

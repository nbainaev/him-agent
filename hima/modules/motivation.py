#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
from hima.common.sdr import SparseSdr
from hima.common.utils import update_exp_trace
from hima.common.utils import softmax
from hima.common.config_utils import TConfig


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
            potential_pct: float, seed: int, connected_pct: float,
            stimulus_threshold: float, active_neurons: int, syn_increment: float,
            syn_decrement: float, output_temperature: float,
            dopa_config: TConfig, dopamine_strength: float
    ):
        self.input_size = input_size
        self.spn_size = output_size * field_size
        self.output_size = output_size
        self.field_size = field_size

        self.stimulus_threshold = stimulus_threshold
        self.dopamine_strength = dopamine_strength
        self.output_temperature = output_temperature
        self.n_active_neurons = active_neurons
        self.syn_increment = syn_increment
        self.syn_decrement = syn_decrement
        self.syn_threshold = synapse_threshold
        self._rng = np.random.default_rng(seed)

        # connections from input to spn neurons
        n_potential_cells = int(potential_pct * input_size)
        self.spn_synapses = np.empty((self.spn_size, n_potential_cells), dtype=int)
        self.spn_synapses_permanence = np.empty((self.spn_size, n_potential_cells), dtype=float)
        self._connected_synapses = np.zeros((self.spn_size, n_potential_cells), dtype=bool)
        for cell in range(self.spn_size):
            potential_cells = self._rng.permutation(input_size)[:n_potential_cells]
            potential_cells.sort()
            for syn, presyn_cell in enumerate(potential_cells):
                if self._rng.uniform() < connected_pct:
                    permanence = self._rng.uniform(synapse_threshold, 1)
                else:
                    permanence = self._rng.uniform(0, synapse_threshold)
                self.spn_synapses[cell, syn] = presyn_cell
                self.spn_synapses_permanence[cell, syn] = permanence

        # connections from spn to output neurons
        self.projection_spn = np.arange(self.spn_size).reshape((output_size, field_size))

        # modulation connections
        self.dopamine_module = TDLambda(seed, n_potential_cells * self.spn_size, **dopa_config)
        self.boost_factors = np.ones(self.spn_size)

    @property
    def spn_connected_syn(self):
        self._connected_synapses.fill(0)
        self._connected_synapses[self.spn_synapses_permanence > self.syn_threshold] = 1
        return self._connected_synapses

    @property
    def spn_excitability(self):
        dop_factors = self.dopamine_module.value_network.cell_value.reshape((
            self.spn_size, -1
        )).copy()
        thr = np.median(dop_factors)
        dop_factors = np.exp(self.dopamine_strength * (dop_factors - thr))
        return dop_factors

    def compute(self, sdr: SparseSdr, reward: float, learn: bool) -> int:
        # compute spn activity
        potential_activation = np.isin(self.spn_synapses, sdr)
        spn_exc = self.spn_excitability
        activity = potential_activation * spn_exc * self.spn_connected_syn
        activity = np.sum(activity, axis=1)

        spn = activity * self.boost_factors
        theta = self.stimulus_threshold * spn_exc.mean()

        k = self.spn_size - self.n_active_neurons
        spn_out = np.argpartition(spn, k)[k:]
        possible_output = np.flatnonzero(spn > theta)
        spn_out = np.intersect1d(spn_out, possible_output)

        if learn:
            incr_cells = potential_activation[spn_out, :]
            decr_cells = np.invert(potential_activation)[spn_out, :]
            self.spn_synapses_permanence[spn_out, :] += self.syn_increment * incr_cells
            self.spn_synapses_permanence[spn_out, :] -= self.syn_increment * decr_cells
            self.spn_synapses_permanence[self.spn_synapses_permanence < 0] = 0
            self.spn_synapses_permanence[self.spn_synapses_permanence > 1] = 1

        values = spn.reshape((self.output_size, -1)) * np.isin(self.projection_spn, spn_out)
        values = values.sum(axis=1)
        probs = softmax(values, self.output_temperature)
        action = self._rng.choice(self.output_size, 1, p=probs)[0]

        if learn:
            spn_upd = np.intersect1d(
                np.arange(action * self.field_size, (action + 1) * self.field_size), spn_out
            )
            not_upd = np.setdiff1d(np.arange(self.spn_size), spn_upd)
            inp_upd = np.isin(self.spn_synapses, sdr)
            inp_upd[not_upd] = False
            upd_sdr = np.flatnonzero(inp_upd)
            self.dopamine_module.update(upd_sdr, reward)

        return action

    def reset(self):
        self.dopamine_module.reset()

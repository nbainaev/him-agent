#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
from hima.common.sdr import SparseSdr
from hima.common.utils import softmax
from hima.common.config_utils import TConfig
from hima.modules.td_lambda import TDLambda


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

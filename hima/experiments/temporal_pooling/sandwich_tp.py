#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.sdr import SDR


class SandwichTp:
    def __init__(
            self, seed: int,
            pooling_decay: float,
            only_upper: bool,
            max_intermediate_used: int = None,
            pooling_decay_r: float = 1.,
            **kwargs
    ):
        rng = np.random.default_rng(seed)

        self.initial_pooling = kwargs['initial_pooling']
        self.pooling_decay = pooling_decay
        self.pooling_decay_r = pooling_decay_r
        self.max_intermediate_used = max_intermediate_used

        self.only_upper = only_upper
        if not self.only_upper:
            self.lower_sp = SpatialPooler(seed=rng.integers(100000), **kwargs['lower_sp_conf'])

        self.upper_sp = SpatialPooler(seed=rng.integers(100000), **kwargs['upper_sp_conf'])

        self._unionSDR = SDR(kwargs['upper_sp_conf']['columnDimensions'])
        self._unionSDR.dense = np.zeros(kwargs['upper_sp_conf']['columnDimensions'])
        self._pooling_activations = np.zeros(kwargs['upper_sp_conf']['inputDimensions'])

        if self.pooling_decay_r > 1:
            self.pooling_decay_mx = self._make_decay_matrix(rng)

        # FIXME: hack to get SandwichTp compatible with other TPs
        # maximum TP active output cells
        self._maxUnionCells = int(
            self.upper_sp.getNumColumns() * self.upper_sp.getLocalAreaDensity()
        )
        self._input_representation = SDR(self._pooling_activations.shape)

    def _pooling_decay_step(self):
        active_pooling_mask = self._pooling_activations != 0
        if self.pooling_decay_r > 1:
            decay_mx = self.pooling_decay_mx
            self._pooling_activations[active_pooling_mask] -= decay_mx[active_pooling_mask]
        else:
            self._pooling_activations[active_pooling_mask] -= self.pooling_decay

        np.clip(self._pooling_activations, 0, 1, out=self._pooling_activations)

    def compute(self, active_neurons: SDR, predicted_neurons: SDR, learn: bool = True) -> SDR:
        self._pooling_decay_step()

        input_representation = self._input_representation
        if not self.only_upper:
            self.lower_sp.compute(predicted_neurons, learn=learn, output=input_representation)
        else:
            input_representation.sparse = np.copy(predicted_neurons.sparse)

        intermediate_sdr = input_representation.sparse
        self._pooling_activations[intermediate_sdr] = (
                self._pooling_activations[intermediate_sdr] + self.initial_pooling
        ).clip(0, 1)

        if self.max_intermediate_used is None:
            # usual behavior
            sdr_to_upper = np.flatnonzero(self._pooling_activations)
        else:
            # restrict used active pooled cells
            top_k = self.max_intermediate_used
            threshold = np.partition(self._pooling_activations, kth=-top_k)[-top_k]
            if threshold == 0:
                sdr_to_upper = np.flatnonzero(self._pooling_activations)
            else:
                sdr_to_upper = np.flatnonzero(self._pooling_activations >= threshold)

        input_representation.sparse = sdr_to_upper
        self.upper_sp.compute(input_representation, learn=learn, output=self._unionSDR)

        return self.getUnionSDR()

    def getUnionSDR(self):
        # ---- middle layer --------

        # res = SDR(self._pooling_activations.shape)
        # res.dense = self._pooling_activations != 0
        # return res
        # --------------------------
        return self._unionSDR

    def getNumInputs(self):
        return self.lower_sp.getNumInputs()

    def getNumColumns(self):
        return self.upper_sp.getNumColumns()

    def reset(self):
        self._pooling_activations = np.zeros(self._pooling_activations.shape)
        self._unionSDR = SDR(self._unionSDR.dense.shape)
        self._unionSDR.dense = np.zeros(self._unionSDR.dense.shape)

    @property
    def output_sdr_size(self):
        return self.upper_sp.getNumColumns()

    @property
    def n_active_bits(self):
        return self._maxUnionCells

    def _make_decay_matrix(self, rng):
        decay, r = self.pooling_decay, self.pooling_decay_r
        decay_mx = decay * loguniform(rng, r, size=self._pooling_activations.size)
        assert (decay_mx/decay).max() <= r and (decay_mx * decay).max() <= r

        return decay_mx


def loguniform(rng, radius, size):
    low = np.log(1 / radius)
    high = np.log(radius)
    return np.exp(rng.uniform(low, high, size))

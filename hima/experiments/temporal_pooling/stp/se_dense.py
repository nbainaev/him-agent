#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import jit
from numpy.random import Generator

from hima.common.config.base import TConfig
from hima.common.sdr import (
    SparseSdr, DenseSdr, OutputMode
)
from hima.common.sds import Sds
from hima.common.timer import timed
from hima.experiments.temporal_pooling.stp.pruning_controller_dense import PruningController
from hima.experiments.temporal_pooling.stp.se import (
    LearningPolicy
)
from hima.experiments.temporal_pooling.stp.se_utils import (
    sample_weights, WeightsDistribution,
    pow_x, dot_match, norm_p, min_match
)


class SpatialEncoderDenseBackend:
    """
    A dense weights for the spatial encoder.
    """
    rng: Generator

    # input
    feedforward_sds: Sds
    adapt_to_ff_sparsity: bool

    # connections
    weights: npt.NDArray[float]
    lebesgue_p: float

    radius: npt.NDArray[float]
    pos_log_radius: npt.NDArray[float]

    pruning_controller: PruningController | None
    pruned_mask: npt.NDArray[bool] | None
    rf: npt.NDArray[int] | None
    rf_sparsity: float

    # potentiation and learning
    learning_policy: LearningPolicy
    # [M, Q]: the number of neurons affected by hebb and anti-hebb
    learning_set: tuple[int, int]
    learning_rate: float

    # output
    output_sds: Sds
    output_mode: OutputMode
    activation_threshold: tuple[float, float, float]

    def __init__(
            self, *, seed: int, feedforward_sds: Sds, output_sds: Sds,

            adapt_to_ff_sparsity,

            lebesgue_p: float = 1.0, init_radius: float = 10.0,
            weights_distribution: WeightsDistribution = WeightsDistribution.NORMAL,
            inhibitory_ratio: float = 0.0,

            match_p: float = 1.0, match_op: str = 'mul',

            learning_policy: LearningPolicy = LearningPolicy.LINEAR, persistent_signs: bool = True,
            normalize_dw: bool = False,

            pruning: TConfig = None,
    ):
        self.rng = np.random.default_rng(seed)

        self.feedforward_sds = Sds.make(feedforward_sds)
        self.output_sds = Sds.make(output_sds)

        self.adapt_to_ff_sparsity = adapt_to_ff_sparsity

        # ==> Weights initialization
        n_out, n_in = self.output_sds.size, self.feedforward_sds.size
        w_shape = (n_out, n_in)

        self.lebesgue_p = lebesgue_p
        self.rf_sparsity = 1.0
        self.has_inhibitory = inhibitory_ratio > 0.0
        assert not self.has_inhibitory, 'Proper support of inhibitory connections is not done yet'
        self.weights = sample_weights(
            self.rng, w_shape, weights_distribution, init_radius, self.lebesgue_p,
            inhibitory_ratio=inhibitory_ratio
        )

        self.radius = self.get_radius()
        self.pos_log_radius = self.get_pos_log_radius()

        # ==> Pattern matching
        self.match_p = match_p
        self.weights_pow_p = None
        if self.match_p != 1.0:
            self.weights_pow_p = self.get_weight_pow_p()

        can_use_min_operator = (
            self.match_p == 1.0 and not self.has_inhibitory
            and learning_policy == LearningPolicy.LINEAR
        )
        if match_op == 'min' and can_use_min_operator:
            self.match_op = min_match
        else:
            match_op = 'mul'
            self.match_op = dot_match

        # ==> Learning
        self.learning_policy = learning_policy
        self.normalize_dw = normalize_dw
        # WARNING: this feature is disabled
        self.persistent_signs = persistent_signs

        # ==> Output
        self.pruning_controller = None
        if pruning is not None:
            self.pruning_controller = PruningController(self, **pruning)
        self.pruned_mask = np.zeros_like(self.weights, dtype=bool)
        self.rf = np.tile(
            np.arange(self.ff_size, dtype=int),
            (self.output_size, 1)
        )

        print(
            f'Init SE dense backend: {self.lebesgue_p}-norm | {self.avg_radius:.3f}'
            f' | {self.learning_policy} | match W^{self.match_p} op: {match_op}'
            f' | Inh: {inhibitory_ratio}'
        )

    def match_input(self, x):
        w = self.weights if self.match_p == 1.0 else self.weights_pow_p
        return self.match_op(x, w)

    def update_weights(self, x, sdr, y, u, lr):
        if sdr.size == 0:
            return

        rf = None if self.rf_sparsity == 1.0 else self.rf

        # TODO: negative Xs is not supported ATM
        if self.learning_policy == LearningPolicy.KROTOV:
            oja_krotov_update(self.weights, sdr, x, u, y, lr, rf)
        else:
            willshaw_update(self.weights, sdr, x, y, lr, rf)

        if self.match_p != 1.0:
            self.weights_pow_p[sdr] = self.get_weight_pow_p(sdr)

        self.radius[sdr] = self.get_radius(sdr)
        self.pos_log_radius[sdr] = self.get_pos_log_radius(sdr)

    def prune_newborns(self, ticks_passed: int = 1):
        pc = self.pruning_controller
        if pc is None or not pc.is_newborn_phase:
            return False
        if not pc.scheduler.tick(ticks_passed):
            return False

        (sparsity, rf_size), t = timed(pc.shrink_receptive_field)(self.pruned_mask)
        # update alive connections
        self.rf = np.array([
            np.flatnonzero(~neuron_connections)
            for neuron_connections in self.pruned_mask
        ])

        stage = pc.stage
        sparsity_pct = round(100.0 * sparsity, 1)
        t = round(t * 1000.0, 2)
        print(f'Prune #{stage}: {sparsity_pct:.1f}% | {rf_size} | {t} ms')

        self.rf_sparsity = sparsity
        # rescale weights to keep the same norms
        old_radius, new_radius = self.radius, self.get_radius()
        # I keep pow weights the same — each will be updated on its next learning step.
        self.weights *= np.expand_dims(old_radius / new_radius, -1)
        return True

    def get_radius(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        p = self.lebesgue_p
        w = self.weights if ixs is None else self.weights[ixs]
        return norm_p(w, p, self.has_inhibitory)

    def get_weight_pow_p(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        p = self.match_p
        w = self.weights if ixs is None else self.weights[ixs]
        return pow_x(w, p, self.has_inhibitory)

    def get_pos_log_radius(self, ixs: npt.NDArray[int] = None) -> npt.NDArray[float]:
        r = self.radius if ixs is None else self.radius[ixs]
        return np.log2(np.maximum(r, 1.0))

    def plot_weights_distr(self):
        import seaborn as sns
        from matplotlib import pyplot as plt
        w, r = self.weights, self.radius
        w = w / np.expand_dims(r, -1)
        p = self.match_p
        w = pow_x(w, p, self.has_inhibitory)
        sns.histplot(w.flatten())
        plt.show()

    @property
    def avg_radius(self):
        return self.radius.mean()

    @property
    def ff_size(self):
        return self.feedforward_sds.size

    @property
    def output_size(self):
        return self.output_sds.size


@jit()
def willshaw_update(weights, sdr, x, y, lr, alive_connections):
    # Willshaw learning rule, L1 normalization:
    # dw = lr * y * (x - w)

    v = y * lr
    for ix, vi in zip(sdr, v):
        w = weights[ix]

        if alive_connections is None:
            w += vi * (x - w)
            fix_anti_hebbian_negatives(vi, w, None)
        else:
            m = alive_connections[ix]
            w[m] += vi * (x[m] - w[m])
            fix_anti_hebbian_negatives(vi, w, m)


@jit()
def oja_krotov_update(weights, sdr, x, u, y, lr, alive_connections):
    # Oja-Krotov learning rule, L^p normalization, p >= 2:
    # dw = lr * y * (x - u * w)

    # NB: u sign persistence is not supported ATM
    # NB2: it also replaces dw normalization
    v = y * lr
    alpha = _get_scale(u)
    if alpha > 1.0:
        v /= alpha

    for ix, vi in zip(sdr, v):
        ui = u[ix]
        w = weights[ix]

        if alive_connections is None:
            w += vi * (x - ui * w)
            fix_anti_hebbian_negatives(vi, w, None)
        else:
            m = alive_connections[ix]
            w[m] += vi * (x[m] - ui * w[m])
            fix_anti_hebbian_negatives(vi, w, m)



@jit()
def willshaw_kuderov_update(weights, sdr, x, y, lr):
    # Willshaw-Kuderov learning rule, sign persistence, L1 normalization:
    # dw = lr * y * [sign(w) * x - w] = lr * y * sign(w) * (x - |w|)

    v = y * lr
    for ix, vi in zip(sdr, v):
        w = weights[ix]
        w += vi * (np.sign(w) * x - w)


@jit()
def oja_krotov_kuderov_update(weights, sdr, x, u, y, lr):
    # Oja-Krotov-Kuderov learning rule, L^p normalization, p >= 2, sign persistence:
    # dw = lr * y * (sign(w) * x - w * u)

    # NB: u sign persistence is not supported ATM
    # NB2: it also replaces dw normalization
    v = y * lr
    alpha = _get_scale(u)
    if alpha > 1.0:
        v /= alpha

    for ix, vi in zip(sdr, v):
        ui = u[ix]
        w = weights[ix]
        w += vi * (np.sign(w) * x - ui * w)


@jit()
def fix_anti_hebbian_negatives(vi, w, mask):
    # `vi` here is a value indicating hebbian (>0) or anti-hebbian (<0)
    if vi >= 0:
        return

    # only anti-hebbian learning causes sign change, if input x is always positive
    # in this case, restrict updates
    if mask is None:
        np.fmax(w, 0.0, w)
    else:
        w[mask] = np.fmax(w[mask], 0.0)

@jit()
def _get_scale(u):
    u_max = np.max(np.abs(u))
    return 1.0 if u_max < 1.0 else u_max ** 0.75 if u_max < 100.0 else u_max ** 0.9

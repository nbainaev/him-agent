#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from __future__ import annotations

from enum import Enum, auto

import numpy as np
from numba import jit
from numpy import typing as npt
from numpy.random import Generator

from hima.common.sdr_array import SdrArray


class WeightsDistribution(Enum):
    NORMAL = 1
    UNIFORM = auto()


def sample_weights(
        rng, w_shape, distribution, radius, lebesgue_p, inhibitory_ratio=0.5
):
    if distribution == WeightsDistribution.NORMAL:
        init_std = get_normal_std(w_shape[1], lebesgue_p, radius)
        weights = np.abs(rng.normal(loc=0., scale=init_std, size=w_shape))

    elif distribution == WeightsDistribution.UNIFORM:
        init_std = get_uniform_std(w_shape[1], lebesgue_p, radius)
        weights = rng.uniform(0., init_std, size=w_shape)

    else:
        raise ValueError(f'Unsupported distribution: {distribution}')

    # make a portion of weights negative
    if inhibitory_ratio > 0.0:
        inh_mask = rng.binomial(1, inhibitory_ratio, size=w_shape).astype(bool)
        weights[inh_mask] *= -1.0

    return weights


def get_uniform_std(n_samples, p, required_r) -> float:
    alpha = 2 / n_samples
    alpha = alpha ** (1 / p)
    return required_r * alpha


def get_normal_std(n_samples: int, p: float, required_r) -> float:
    alpha = np.pi / (2 * n_samples)
    alpha = alpha ** (1 / p)
    return required_r * alpha


def arg_top_k(x, k):
    k = min(k, x.size)
    return np.argpartition(x, -k, axis=-1)[..., -k:]


def boosting(
        relative_rate: float | npt.NDArray[float], k: float | npt.NDArray[float],
        softness: float = 3.0
) -> float:
    # relative rate: rate / R_target
    # x = -log(relative_rate)
    #   0 1 +inf  -> +inf 0 -inf
    x = -np.log(relative_rate)

    # relative_rate -> x -> B:
    #   0 -> +inf -> K^tanh(+inf) = K
    #   1 -> 0 -> K^tanh(0) = 1
    #   +inf -> -inf -> K^tanh(-inf) = 1 / K
    # higher softness just makes the sigmoid curve smoother; default value is empirically optimized
    # TODO: check addition, it could be too high -> 1.05
    return np.power(k + 1.1, np.tanh(x / softness))


@jit
def nb_choice(rng, p):
    """Choose a sample from N values with weights p (they could be non-normalized)."""
    # Get cumulative weights
    acc_w = np.cumsum(p)
    # Total of weights
    mx_w = acc_w[-1]
    r = mx_w * rng.random()
    # Get corresponding index
    ind = np.searchsorted(acc_w, r, side='right')
    return ind


@jit()
def nb_choice_k(
        rng: Generator, k: int, weights: npt.NDArray[np.float64] = None, n: int = None,
        replace: bool = False, cache: npt.NDArray[np.bool_] = None
):
    """
    Choose k samples from max_n values, with optional weights and replacement.

    Be careful using this function with very skewed weights w/o replacement, as it may be slow and
    end up with time limit exception. For such cases, consider using numpy choice with weights.
    The reason for this is that when probability mass is concentrated in a few values,
    the random sampling will repeatedly produce duplicates, which are then rejected
    due to the no-replacement constraint. So, the best case scenario is when the weights are
    close to uniform.
    """
    acc_w = np.cumsum(weights) if weights is not None else np.arange(0, n, 1, dtype=np.float64)
    # Total of weights
    mx_w = acc_w[-1]
    # result
    result = np.full(k, -1, dtype=np.int64)
    if not replace and cache is None and n is not None:
        cache = np.zeros(n, dtype=np.bool_)

    i, j = 0, 0
    # reasonable time limit: avg 100 tries for each required sample, growing as the
    # choice getting more dense (n / k -> 1)
    timelimit = min(10_000_000, k * 100 ** (1.0 + k / n))
    while i < k and j < timelimit:
        j += 1
        r = mx_w * rng.random()
        ind = np.searchsorted(acc_w, r, side='right')

        if not replace and cache[ind]:
            continue
        else:
            result[i] = ind
            if not replace:
                cache[ind] = True
            i += 1

    if j >= timelimit:
        # print(f'{j=}: {i=} of {k=} in {n=} | {mx_w=}')
        # with np.printoptions(precision=3):
        #     print(f'{acc_w[:20]=}')
        #     print(f'{acc_w[-20:]=}')
        raise ValueError('Infinite loop in nb_choice_k. If weights are degenerate, use numpy')

    return result


def pow_x(x, p, has_negative):
    if p == 1.0:
        return x

    if has_negative:
        return np.sign(x) * (np.abs(x) ** p)
    return x ** p


def abs_pow_x(x, p, has_negative):
    if has_negative:
        x = np.abs(x)
    if p == 1.0:
        return x
    return x ** p


def dot_match(x, w):
    return np.dot(w, x.T).T


def min_match(x, w):
    inter = np.empty_like(w)
    return np.vstack([
        np.sum(np.fmin(w, xx, out=inter), -1)
        for xx in x
    ])


@jit()
def min_match_j(x, w):
    n_batch = x.shape[0]
    n_out, n_in = w.shape
    res = np.zeros((n_batch, n_out))
    for k in range(n_batch):
        for i in range(n_out):
            t = 0
            for xx, ww in zip(x[k], w[i]):
                t += min(xx, ww)
            res[k, i] = t
    return res


def dot_match_sparse(x: SdrArray, w, ixs_srt_j, shifts, srt_i):
    batch_size = len(x)
    n_out = srt_i.shape[0]

    res = np.zeros((batch_size, n_out))
    for i in range(batch_size):
        match_sparse_w_x_(
            res[i, :], w, ixs_srt_j, shifts, x.sparse[i].sdr, x.sparse[i].values
        )
    return res


@jit()
def match_sparse_w_x_(res, wi_sp, jxs, shifts, sdr, rates):
    for i, r in zip(sdr, rates):
        for j in range(shifts[i], shifts[i + 1]):
            res[jxs[j]] += r * wi_sp[j]
    return res


def norm_p(x, p, has_negative):
    return abs_pow_x(
        np.sum(abs_pow_x(x, p, has_negative), -1),
        (1 / p),
        # even if they were negative, already took abs
        False
    )


def normalize(x, p=1.0, has_negative=False):
    r = norm_p(x, p, has_negative)
    eps = 1e-30
    if x.ndim > 1:
        mask = r > eps
        res = x.copy()
        res[mask] /= np.expand_dims(r[mask], -1)
        return res

    return x / r if r > eps else x

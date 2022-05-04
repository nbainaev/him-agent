#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np

from hima.common.sdr import DenseSdr, SparseSdr
from hima.common.utils import safe_divide
from htm.bindings.sdr import SDR


def symmetric_diff_sz(set1: DenseSdr, set2: DenseSdr) -> int:
    return np.setdiff1d(set1, set2).size + np.setdiff1d(set2, set1).size


def symmetric_error(_output, _target):
    if _output.size + _target.size == 0:
        return 0
    return symmetric_diff_sz(_output, _target) / np.union1d(_output, _target).size


def representations_intersection_1(dense1, dense2):
    if np.count_nonzero(dense1) == 0:
        return 1
    return np.count_nonzero(dense1 * dense2) / np.count_nonzero(dense1)


def row_similarity(policy_1, policy_2):
    counter = 0
    for index in range(len(policy_1)):
        if policy_1[index] == policy_2[index]:
            counter += 1
    return counter / len(policy_1)


def representation_similarity(representation_1, representation_2):
    overlap = np.count_nonzero(representation_1 * representation_2)
    union = np.count_nonzero(representation_1 | representation_2)
    if union == 0:
        return 1
    return overlap / union


def sdrs_similarity(sdr1: SDR, sdr2: SDR):
    intersection = np.intersect1d(sdr1.sparse, sdr2.sparse).shape[0]
    union = np.union1d(sdr1.sparse, sdr2.sparse).shape[0]
    return safe_divide(intersection, union, 1)


def similarity_mae(pure, representational):
    return np.mean(abs(pure - representational)[np.ones(pure.shape) - np.identity(pure.shape[0]) == 1])


def sdr_similarity(x1: set, x2: set, symmetrical=False) -> float:
    overlap = len(x1 & x2)
    if symmetrical:
        return safe_divide(overlap, len(x1 | x2))
    return safe_divide(overlap, len(x2))


def tuple_similarity(t1: tuple[SparseSdr, ...], t2: tuple[SparseSdr, ...]) -> float:
    sim = 1.
    for i in range(len(t1)):
        sim *= sdr_similarity(t1[i], t2[i])
    return sim


def entropy(x: np.ndarray) -> float:
    return -np.nansum(x * np.log(x))


def mean_absolute_error(x: np.ndarray, y: np.ndarray) -> float:
    # noinspection PyTypeChecker
    return np.mean(np.abs(x - y))

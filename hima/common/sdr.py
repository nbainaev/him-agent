#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from typing import Iterable, Union, Sequence

import numpy as np

SparseSdr = Union[Iterable[int], Sequence[int], np.ndarray]
DenseSdr = np.ndarray


def sparse_to_dense(indices: SparseSdr, total_size: int) -> DenseSdr:
    """Converts SDR from sparse representation to dense."""
    dense_vector = np.zeros(total_size, dtype=np.int8)
    dense_vector[indices] = 1
    return dense_vector


def dense_to_sparse(dense_vector: DenseSdr) -> SparseSdr:
    return np.nonzero(dense_vector)[0]


class Sds:
    """Sparse Distributed Space parameters"""
    shape: tuple[int, ...]
    size: int
    sparsity: float
    active_size: int

    def __init__(
            self,
            short_notation: tuple[Union[tuple, int, float], Union[int, float]] = None,
            *,
            shape: tuple[int, ...] = None,
            size: int = None,
            sparsity: float = None,
            active_size: int = None,
    ):
        if short_notation is not None:
            # ignore keyword-only params
            shape, size, sparsity, active_size = self.parse_short_notation(*short_notation)

        self.shape, self.size, self.sparsity, self.active_size = self.resolve_all(
            shape=shape, size=size, sparsity=sparsity, active_size=active_size
        )

    @staticmethod
    def parse_short_notation(first, second):
        shape, size, sparsity, active_size = None, None, None, None

        if isinstance(first, tuple) or isinstance(first, int):
            # (shape/size, sparsity/active_size)

            if isinstance(first, tuple):
                # shape
                shape = first
            else:
                # size
                size = first

            if isinstance(second, float):
                # sparsity
                sparsity = second
            elif isinstance(second, int):
                # active size
                active_size = second
            else:
                raise TypeError(
                    f'Parsed as (shape/size, sparsity/active_size). TSecond: {type(second)}'
                )
        elif isinstance(first, float):
            # (sparsity, active_size)
            if not isinstance(second, int):
                raise TypeError(
                    f'Parsed as (sparsity, active_size). TSecond is not int: {type(second)}'
                )
            sparsity = first
            active_size = second
        else:
            raise TypeError(f'TFirst is not tuple/int/float: {type(first)}')

        return shape, size, sparsity, active_size

    @staticmethod
    def resolve_all(
            shape: tuple[int, ...] = None,
            size: int = None,
            sparsity: float = None,
            active_size: int = None
    ):
        if shape is not None or size is not None:
            # shape and/or size is defined
            # and at least sparsity or active size should be defined too

            if size is None:
                size = np.prod(shape)
            elif shape is None:
                shape = (size,)

            # will raise error if both None
            if active_size is None:
                active_size = int(size * sparsity)
            elif sparsity is None:
                sparsity = active_size / size
        else:
            # shape and size both are NOT defined
            # sparsity and active size both should be defined
            size = round(active_size / sparsity)
            shape = (size, )

        return shape, size, sparsity, active_size

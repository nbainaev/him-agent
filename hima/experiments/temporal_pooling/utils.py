#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
from htm.bindings.sdr import SDR


class StupidEncoder:
    def __init__(self, _n_actions, output_size):
        self.n_actions = _n_actions
        self.action_size = int(output_size / _n_actions)
        self.output_size = output_size

    def encode(self, action: int):
        result = np.zeros(self.output_size)
        result[self.action_size * action: self.action_size * (action + 1)] = 1
        return result


class IdentityEncoder:
    @staticmethod
    def encode(state):
        return state


def make_sdr(pos: tuple, _shape: tuple) -> SDR:
    result = SDR(_shape)
    numpy_res = result.dense
    numpy_res = np.zeros(_shape)
    numpy_res[pos] = 1
    result.dense = numpy_res
    return result


def make_sdrs(array: np.ndarray, _shape: tuple) -> np.ndarray:
    result = np.ndarray((array.size,), dtype=SDR)
    iterator = 0
    for number in array:
        result[iterator] = make_sdr(number, _shape)
        iterator += 1
    return result


def get_one_hot(num, _classes_num=25):
    _data = np.zeros(_classes_num)
    _data[num] = 1
    return _data


def one_hot(num, _classes_num=25):
    _data = np.zeros(_classes_num)
    _data[num] = 1
    return _data
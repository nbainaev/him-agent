#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
from hima.envs.mpg import MarkovProcessGrammar


def main():
    # process from "Deep Predictive Learning in Neocortex and Pulvinar" O'Reilly 2021
    transitions = [
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0.5, 0.5, 0, 0, 0, 0],
        [0, 0, 0.5, 0, 0, 0.5, 0, 0],
        [0, 0, 0, 0.5, 0.5, 0, 0, 0],
        [0, 0, 0.5, 0, 0, 0, 0.5, 0],
        [0, 0, 0, 0, 0.5, 0, 0.5, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]

    letters = [
        [0, 'B', 0, 0, 0, 0, 0, 0],
        [0, 0, 'P', 'T', 0, 0, 0, 0],
        [0, 0, 'T', 0, 0, 'V', 0, 0],
        [0, 0, 0, 'S', 'X', 0, 0, 0],
        [0, 0, 'X', 0, 0, 0, 'S', 0],
        [0, 0, 0, 0, 'P', 0, 'V', 0],
        [0, 0, 0, 0, 0, 0, 0, 'E'],
    ]

    mpg = MarkovProcessGrammar(
        8,
        transitions,
        letters,
        initial_state=0,
        terminal_state=7,
        autoreset=False,
        seed=432
    )

    for i in range(10):
        mpg.reset()
        word = []
        while True:
            letter = mpg.next_state()

            if letter:
                word.append(letter)
            else:
                break

        print(i, ''.join(word))


if __name__ == '__main__':
    main()

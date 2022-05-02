#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
from hima.envs.mpg import MarkovProcessGrammar
from hima.common.sdr_encoders import IntBucketEncoder
from hima.modules.htm.belief_tm import NaiveBayesTM

import wandb


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

    char_to_number = {'B': 0, 'P': 1, 'T': 2, 'V': 3, 'S': 4, 'X': 5, 'E': 6}

    log = True
    bucket_size = 3
    seed = 6564
    encoder = IntBucketEncoder(len(char_to_number), bucket_size)

    mpg = MarkovProcessGrammar(
        8,
        transitions,
        letters,
        initial_state=0,
        terminal_state=7,
        autoreset=False,
        seed=seed
    )
    print(f'n_columns: {encoder.output_sdr_size}')
    tm = NaiveBayesTM(
        encoder.output_sdr_size,
        cells_per_column=8,
        max_segments_per_cell=3,
        max_receptive_field_size=-1,
        w_lr=0.1,
        w_punish=0.1,
        theta_lr=0.1,
        b_lr=0.1,
        seed=seed
    )
    if log:
        logger = wandb.init(
            project='test_belief_tm', entity='hauska',
            config=dict(
                seed=seed,
                bucket_size=bucket_size
            )
        )
    else:
        logger = None

    for i in range(1000):
        mpg.reset()
        tm.reset()

        word = []
        surprise = []
        anomaly = []

        while True:
            letter = mpg.next_state()

            if letter:
                word.append(letter)
            else:
                break

            tm.set_active_columns(encoder.encode(char_to_number[letter]))
            tm.activate_cells(learn=True)

            surprise.append(min(200.0, tm.surprise))
            anomaly.append(tm.anomaly)

            tm.activate_dendrites()
            tm.predict_cells()
        if logger is not None:
            logger.log({
                'surprise': np.array(surprise)[1:].mean(),
                'anomaly': np.array(anomaly)[1:].mean()
            })

    ...


if __name__ == '__main__':
    main()

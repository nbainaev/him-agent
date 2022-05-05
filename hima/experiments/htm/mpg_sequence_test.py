#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
from hima.envs.mpg import MarkovProcessGrammar
from hima.common.sdr_encoders import IntBucketEncoder
from hima.modules.htm.belief_tm import NaiveBayesTM
from hima.modules.htm.temporal_memory import ClassicTemporalMemory
from htm.bindings.sdr import SDR

import wandb
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import yaml
import os


def main(config_path):
    config = dict()

    with open(config_path, 'r') as file:
        config['run'] = yaml.load(file, Loader=yaml.Loader)

    with open(config['run']['tm_conf'], 'r') as file:
        config['tm'] = yaml.load(file, Loader=yaml.Loader)

    with open(config['run']['mpg_conf'], 'r') as file:
        config['mpg'] = yaml.load(file, Loader=yaml.Loader)

    mpg = MarkovProcessGrammar(
        seed=config['run']['seed'],
        **config['mpg']
    )

    encoder = IntBucketEncoder(len(mpg.alphabet), config['run']['bucket_size'])

    print(f'n_columns: {encoder.output_sdr_size}')

    if config['run']['log']:
        logger = wandb.init(
            project=config['run']['project_name'], entity=os.environ['WANDB_ENTITY'],
            config=config
        )
    else:
        logger = None

    if config['run']['tm_type'] == 'naive_bayes':
        run_naive_bayes(config, mpg, encoder, logger)
    elif config['run']['tm_type'] == 'classic':
        run_classic_tm(config, mpg, encoder, logger)


def run_naive_bayes(config, mpg, encoder, logger):
    tm = NaiveBayesTM(
        encoder.output_sdr_size,
        seed=config['run']['seed'],
        **config['tm']
    )

    density = np.zeros((8, 7))
    lr = 0.02

    hist_dist = np.zeros((8, 7))

    for i in range(config['run']['epochs']):
        mpg.reset()
        tm.reset()

        word = []
        surprise = []
        anomaly = []
        confidence = []

        while True:
            letter = mpg.next_state()

            if letter:
                word.append(letter)
            else:
                break

            tm.set_active_columns(encoder.encode(mpg.char_to_num[letter]))
            tm.activate_cells(learn=True)

            surprise.append(min(200.0, tm.surprise))
            anomaly.append(tm.anomaly)
            confidence.append(tm.confidence)

            tm.activate_dendrites()
            tm.predict_cells()

            letter_dist = np.prod(tm.column_probs.reshape((-1, config['run']['bucket_size'])).T, axis=0)
            density[mpg.current_state] += lr * (letter_dist - density[mpg.current_state])

            pred_columns_dense = np.zeros(tm.n_columns)
            pred_columns_dense[tm.predicted_columns] = 1
            predicted_letters = np.prod(pred_columns_dense.reshape((-1, config['run']['bucket_size'])).T, axis=0)
            hist_dist[mpg.current_state] += lr * (predicted_letters - hist_dist[mpg.current_state])

        if logger is not None:
            logger.log({
                'surprise': np.array(surprise)[1:].mean(),
                'anomaly': np.array(anomaly)[1:].mean(),
                'confidence': np.array(confidence)[1:].mean()
            }, step=i)

            if i % config['run']['update_rate'] == 0:
                def format_fn(tick_val, tick_pos):
                    if int(tick_val) in range(len(mpg.alphabet)):
                        return mpg.alphabet[int(tick_val)]
                    else:
                        return ''

                fig, ax = plt.subplots(2, sharex=True)
                ax[0].xaxis.set_major_formatter(format_fn)
                ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
                for x in range(density.shape[0]):
                    ax[0].plot(density[x], label=f'state{x}', linewidth=2, marker='o')
                ax[0].grid()

                ax[1].xaxis.set_major_formatter(format_fn)
                ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
                for x in range(hist_dist.shape[0]):
                    ax[1].plot(hist_dist[x], linewidth=2, marker='o')
                ax[1].grid()

                fig.legend(loc=7)

                logger.log({f'letter_predictions': wandb.Image(fig)}, step=i)

                plt.close(fig)

    ...


def run_classic_tm(config, mpg, encoder, logger):
    tm = ClassicTemporalMemory(
        encoder.output_sdr_size,
        n_active_bits=config['run']['bucket_size'],
        seed=config['run']['seed'],
        **config['tm']
    )

    hist_dist = np.zeros((8, 7))
    lr = 0.02

    for i in range(config['run']['epochs']):
        mpg.reset()
        tm.reset()

        word = []
        anomaly = []
        confidence = []
        predicted_columns = np.empty(0)

        while True:
            letter = mpg.next_state()

            if letter:
                word.append(letter)
            else:
                break

            active_columns = SDR(tm.n_columns)
            active_columns.sparse = encoder.encode(mpg.char_to_num[letter])

            tm.compute(active_columns, learn=True)

            anomaly.append(tm.anomaly)

            conf = len(predicted_columns) / len(active_columns.sparse)

            confidence.append(conf)

            tm.activateDendrites(learn=True)
            predicted_cells = tm.getPredictiveCells().sparse
            pred_columns_dense = np.zeros(tm.n_columns)
            if len(predicted_cells) > 0:
                predicted_columns = np.unique(predicted_cells // tm.cells_per_column)
                pred_columns_dense[predicted_columns] = 1
            else:
                predicted_columns = np.empty(0)

            predicted_letters = np.prod(pred_columns_dense.reshape((-1, config['run']['bucket_size'])).T, axis=0)
            hist_dist[mpg.current_state] += lr * (predicted_letters - hist_dist[mpg.current_state])

        if logger is not None:
            logger.log({
                'anomaly': np.array(anomaly)[1:].mean(),
                'confidence': np.array(confidence)[1:].mean()
            }, step=i)

            if i % config['run']['update_rate'] == 0:
                def format_fn(tick_val, tick_pos):
                    if int(tick_val) in range(len(mpg.alphabet)):
                        return mpg.alphabet[int(tick_val)]
                    else:
                        return ''

                fig, ax = plt.subplots()

                ax.xaxis.set_major_formatter(format_fn)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                for x in range(hist_dist.shape[0]):
                    ax.plot(hist_dist[x], label=f'state_{x}', linewidth=2, marker='o')
                ax.grid()

                fig.legend(loc=7)

                logger.log({f'letter_predictions': wandb.Image(fig)}, step=i)

                plt.close(fig)

    ...


if __name__ == '__main__':
    main('configs/mpg_runner.yaml')

#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
from hima.envs.mpg import MultiMarkovProcessGrammar
from hima.common.sdr_encoders import IntBucketEncoder
from hima.modules.htm.belief_tm import HybridNaiveBayesTM
import pickle

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

    mpg = MultiMarkovProcessGrammar(
        seed=config['run']['seed'],
        **config['mpg']
    )

    obs_encoder = IntBucketEncoder(len(mpg.alphabet), config['run']['bucket_size'])
    policy_encoder = IntBucketEncoder(len(mpg.policy_transition_probs), config['run']['bucket_size'])

    print(f'n_columns: {obs_encoder.output_sdr_size}')

    if config['run']['log']:
        logger = wandb.init(
            project=config['run']['project_name'], entity=os.environ['WANDB_ENTITY'],
            config=config
        )
    else:
        logger = None

    if config['run']['tm_type'] == 'hybrid_naive_bayes':
        run_hybrid_naive_bayes_tm(config, mpg, obs_encoder, policy_encoder, logger)
    else:
        raise ValueError


def run_hybrid_naive_bayes_tm(config, mpg, obs_encoder, policy_encoder, logger):
    activation_threshold = int(config['run']['bucket_size']*config['tm'].pop('activation_threshold'))
    learning_threshold = int(config['run']['bucket_size']*config['tm'].pop('learning_threshold'))

    tm = HybridNaiveBayesTM(
        columns=obs_encoder.output_sdr_size,
        context_cells=obs_encoder.output_sdr_size * config['tm']['cells_per_column'],
        feedback_cells=policy_encoder.output_sdr_size,
        activation_threshold_basal=activation_threshold,
        learning_threshold_basal=learning_threshold,
        learning_threshold_inhib=learning_threshold,
        activation_threshold_apical=activation_threshold,
        learning_threshold_apical=learning_threshold,
        seed=config['run']['seed'],
        **config['tm']
    )

    density = np.zeros((policy_encoder.n_values, len(mpg.states), len(mpg.alphabet)))
    hist_dist = np.zeros((policy_encoder.n_values, len(mpg.states), len(mpg.alphabet)))
    lr = 0.02

    for i in range(config['run']['epochs']):
        mpg.reset()
        tm.reset()

        word = []
        anomaly = []
        confidence = []
        surprise = []

        policy = mpg.rng.integers(policy_encoder.n_values)
        mpg.set_policy(policy)

        while True:
            letter = mpg.next_state()

            if letter:
                word.append(letter)
            else:
                break
            # set winner cells from previous step
            tm.set_active_context_cells(tm.get_winner_cells())
            tm.set_active_columns(obs_encoder.encode(mpg.char_to_num[letter]))
            tm.set_active_feedback_cells(policy_encoder.encode(policy))
            tm.activate_cells(learn=True)

            surprise.append(min(200.0, tm.surprise))
            anomaly.append(tm.anomaly[-1])

            conf = len(tm.get_predicted_columns()) / len(tm.get_active_columns())
            confidence.append(conf)

            tm.set_active_context_cells(tm.get_active_cells())
            tm.activate_basal_dendrites(learn=True)
            tm.predict_cells()
            tm.predict_columns()

            letter_dist = np.mean(
                tm.column_probs.reshape((-1, config['run']['bucket_size'])).T, axis=0
            )
            density[policy][mpg.current_state] += lr * (letter_dist - density[policy][mpg.current_state])

            pred_columns_dense = np.zeros(tm.columns)
            pred_columns_dense[tm.get_predicted_columns()] = 1
            predicted_letters = np.mean(
                pred_columns_dense.reshape((-1, config['run']['bucket_size'])).T, axis=0
            )
            hist_dist[policy][mpg.current_state] += lr * (predicted_letters - hist_dist[policy][mpg.current_state])

        if logger is not None:
            logger.log(
                {
                    'surprise': np.array(surprise)[1:].mean(),
                    'anomaly': np.array(anomaly)[1:].mean(),
                    'confidence': np.array(confidence)[1:].mean()
                }, step=i
            )
            for pol in range(density.shape[0]):
                if i % config['run']['update_rate'] == 0:
                    def format_fn(tick_val, tick_pos):
                        if int(tick_val) in range(len(mpg.alphabet)):
                            return mpg.alphabet[int(tick_val)]
                        else:
                            return ''

                    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
                    ax1.xaxis.set_major_formatter(format_fn)
                    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
                    ax1.set_ylim(0, 1)
                    for x in range(density[pol].shape[0]):
                        ax1.plot(density[pol][x], label=f'state{x}', linewidth=2, marker='o')
                    ax1.grid()

                    ax2.xaxis.set_major_formatter(format_fn)
                    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
                    ax2.set_ylim(0, 1)
                    for x in range(hist_dist[pol].shape[0]):
                        ax2.plot(hist_dist[pol][x], linewidth=2, marker='o')
                    ax2.grid()

                    fig.legend(loc=7)

                    logger.log({f'letter_predictions_policy_{pol}': wandb.Image(fig)}, step=i)

                    plt.close(fig)

    if logger is not None:
        name = logger.name
    else:
        name = np.random.bytes(32)

    np.save(f'logs/density_{name}.npy', density)
    np.save(f'logs/hist_dist_{name}.npy', hist_dist)

    if config['run']['save_model']:
        with open(f"logs/model_env_{name}_{config['run']['epochs']}.pkl", 'wb') as file:
            pickle.dump((tm, mpg, obs_encoder, policy_encoder), file)


if __name__ == '__main__':
    main('configs/mpg_runner.yaml')

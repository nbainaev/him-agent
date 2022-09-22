#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.modules.hmm import CHMMBasic
from hima.envs.mpg import MarkovProcessGrammar
import numpy as np
from scipy.special import rel_entr
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import yaml
import os


class HMMRunner:
    def __init__(self, logger, conf):
        self.seed = conf['run']['seed']

        conf['hmm']['seed'] = self.seed
        conf['mpg']['seed'] = self.seed

        self.mpg = MarkovProcessGrammar(**conf['mpg'])

        conf['hmm']['n_columns'] = len(self.mpg.alphabet)
        self.hmm = CHMMBasic(**conf['hmm'])

        self.n_episodes = conf['run']['n_episodes']
        self.smf_dist = conf['run']['smf_dist']
        self.log_update_rate = conf['run']['update_rate']
        self.logger = logger

    def run(self):
        dist = np.zeros((len(self.mpg.states), len(self.mpg.alphabet)))
        true_dist = np.array([self.mpg.predict_letters(from_state=i) for i in self.mpg.states])

        for i in range(self.n_episodes):
            self.mpg.reset()
            self.hmm.reset()

            dkls = []
            surprises = []

            while True:
                prev_state = self.mpg.current_state

                letter = self.mpg.next_state()

                if letter is None:
                    break
                else:
                    obs_state = self.mpg.char_to_num[letter]

                column_probs = self.hmm.predict_columns()
                self.hmm.observe(obs_state, learn=True)

                # metrics
                # 1. surprise
                active_columns = np.arange(self.hmm.n_columns) == obs_state
                surprise = - np.sum(np.log(column_probs[active_columns]))
                surprise += - np.sum(np.log(1 - column_probs[~active_columns]))

                surprises.append(surprise)

                # 2. distribution
                dist[prev_state] += self.smf_dist * (
                        column_probs - dist[prev_state]
                )

                # 3. Kl distance
                dkls.append(
                    min(
                        rel_entr(true_dist[prev_state], column_probs).sum(),
                        200.0
                    )
                )

            if self.logger is not None:
                self.logger.log(
                    {
                        'main_metrics/surprise': np.array(surprises).mean(),
                        'main_metrics/dkl': np.array(np.abs(dkls)).mean(),
                    }, step=i
                )

                if i % self.log_update_rate == 0:
                    kl_divs = rel_entr(true_dist, dist).sum(axis=-1)

                    n_states = len(self.mpg.states)
                    k = int(np.ceil(np.sqrt(n_states)))
                    fig, axs = plt.subplots(k, k)
                    fig.tight_layout(pad=3.0)

                    for n in range(n_states):
                        ax = axs[n // k][n % k]
                        ax.grid()
                        ax.set_ylim(0, 1)
                        ax.set_title(
                            f's: {n}; ' + '$D_{KL}$: ' + f'{np.round(kl_divs[n], 2)}'
                        )
                        ax.bar(
                            np.arange(dist[n].shape[0]),
                            dist[n],
                            tick_label=self.mpg.alphabet,
                            label='TM',
                            color=(0.7, 1.0, 0.3),
                            capsize=4,
                            ecolor='#2b4162'
                        )
                        ax.bar(
                            np.arange(dist[n].shape[0]),
                            true_dist[n],
                            tick_label=self.mpg.alphabet,
                            color='#8F754F',
                            alpha=0.6,
                            label='True'
                        )

                        fig.legend(['Predicted', 'True'], loc=8)

                        self.logger.log(
                            {'density/letter_predictions': wandb.Image(fig)}, step=i
                        )

                        plt.close(fig)

                    self.logger.log(
                        {
                            'weights/priors': wandb.Image(
                                sns.heatmap(
                                    self.hmm.log_state_prior.reshape((1, -1))
                                )
                            )
                        },
                        step=i
                    )
                    plt.close('all')
                    self.logger.log(
                        {
                            'weights/transitions': wandb.Image(
                                sns.heatmap(self.hmm.log_transition_factors)
                            )
                        },
                        step=i
                    )
                    plt.close('all')


def main(config_path):
    config = dict()

    with open(config_path, 'r') as file:
        config['run'] = yaml.load(file, Loader=yaml.Loader)

    with open(config['run']['hmm_conf'], 'r') as file:
        config['hmm'] = yaml.load(file, Loader=yaml.Loader)

    with open(config['run']['mpg_conf'], 'r') as file:
        config['mpg'] = yaml.load(file, Loader=yaml.Loader)

    if config['run']['log']:
        logger = wandb.init(
            project=config['run']['project_name'], entity=os.environ['WANDB_ENTITY'],
            config=config
        )
    else:
        logger = None

    runner = HMMRunner(logger, config)
    runner.run()


if __name__ == '__main__':
    main('configs/hmm_runner.yaml')

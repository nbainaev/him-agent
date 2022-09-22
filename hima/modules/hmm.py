#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np
from typing import Literal, Optional
from hima.common.utils import softmax

L_MODE = Literal['bw', 'mc', 'delta']
INI_MODE = Literal['normal', 'uniform']


class CHMMBasic:
    def __init__(
            self,
            n_columns: int,
            cells_per_column: int,
            lr: float = 0.1,
            temp: float = 1.0,
            regularization: float = 0.1,
            learning_mode: L_MODE = 'mc',
            initialization: INI_MODE = 'uniform',
            seed: Optional[int] = None
    ):
        self.n_columns = n_columns
        self.cells_per_column = cells_per_column
        self.n_states = cells_per_column * n_columns
        self.states = np.arange(self.n_states)
        self.lr = lr
        self.temp = temp
        self.alpha = regularization
        self.learning_mode = learning_mode
        self.initialization = initialization
        self.is_first = True
        self.seed = seed

        self._rng = np.random.default_rng(self.seed)

        if self.initialization == 'normal':
            self.log_transition_factors = self._rng.normal(size=(self.n_states, self.n_states))
        elif self.initialization == 'uniform':
            self.log_transition_factors = np.zeros((self.n_states, self.n_states))

        self.transition_probs = np.vstack(
            [softmax(x, self.temp) for x in self.log_transition_factors]
        )

        self.log_state_prior = np.zeros(self.n_states)
        self.state_prior = softmax(self.log_state_prior, self.temp)
        self.forward_message = self.state_prior
        self.backward_message = np.ones(self.n_states) / self.n_states
        self.active_state = None
        self.prediction = None

    def observe(self, observation_state: int, learn: bool = True) -> None:
        assert 0 <= observation_state < self.n_columns, "Invalid observation state."
        assert self.prediction is not None, "Run predict_columns() first."

        obs_factor = np.zeros(self.n_states)

        states_for_obs = np.arange(
            self.cells_per_column * observation_state,
            self.cells_per_column * (observation_state + 1),
        )
        obs_factor[states_for_obs] = 1

        new_forward_message = self.prediction * obs_factor
        new_forward_message /= np.sum(new_forward_message)

        if learn:
            prev_state = self.active_state

            predicted_state = self._rng.choice(self.states, p=self.prediction)
            next_state = self._rng.choice(self.states, p=new_forward_message)

            wrong_prediction = not np.in1d(predicted_state, states_for_obs)

            if self.is_first:
                w = self.log_state_prior[next_state]
                self.log_state_prior[next_state] += self.lr * (1 - self.alpha * w)
                self.state_prior = softmax(self.log_state_prior, self.temp)

                if wrong_prediction:
                    self.log_state_prior[prev_state] -= self.lr * self.alpha * w

                self.is_first = False
            else:
                w = self.log_transition_factors[prev_state, next_state]
                self.log_transition_factors[prev_state, next_state] += self.lr * (1 - self.alpha * w)

                if wrong_prediction:
                    self.log_transition_factors[prev_state, predicted_state] -= self.lr * self.alpha * w

            self.transition_probs = np.vstack(
                [softmax(x, self.temp) for x in self.log_transition_factors]
            )

            self.active_state = next_state

        self.forward_message = new_forward_message

    def predict_columns(self):
        if self.is_first:
            self.prediction = self.state_prior
        else:
            self.prediction = np.dot(self.forward_message, self.transition_probs)
        prediction = np.reshape(self.prediction, (self.n_columns, self.cells_per_column))
        prediction = prediction.sum(axis=-1)
        return prediction

    def reset(self):
        self.forward_message = self.state_prior
        self.backward_message = np.ones(self.n_states) / self.n_states
        self.active_state = None
        self.prediction = None
        self.is_first = True

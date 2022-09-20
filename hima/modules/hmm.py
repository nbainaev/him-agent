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
            learning_mode: L_MODE = 'mc',
            initialization: INI_MODE = 'uniform',
            seed: Optional[int] = None
    ):
        self.n_columns = n_columns
        self.cells_per_column = cells_per_column
        self.n_states = cells_per_column * n_columns
        self.lr = lr
        self.learning_mode = learning_mode
        self.initialization = initialization
        self.seed = seed

        self._rng = np.random.default_rng(self.seed)

        if self.initialization == 'normal':
            self.log_transition_factors = self._rng.normal(size=(self.n_states, self.n_states))
        elif self.initialization == 'uniform':
            self.log_transition_factors = np.zeros((self.n_states, self.n_states))

        self.transition_probs = np.vstack([softmax(x) for x in self.log_transition_factors])

        self.forward_message = np.ones(self.n_states) / self.n_states
        self.backward_message = np.ones(self.n_states) / self.n_states
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
            states = np.arange(self.n_states)
            predicted_state = self._rng.choice(states, p=self.prediction)
            prev_state = self._rng.choice(states, p=self.forward_message)
            next_state = self._rng.choice(states, p=new_forward_message)

            self.log_transition_factors[prev_state, next_state] += self.lr

            if not np.in1d(predicted_state, states_for_obs):
                self.log_transition_factors[prev_state, predicted_state] -= self.lr

            self.transition_probs = np.vstack([softmax(x) for x in self.log_transition_factors])

        self.forward_message = new_forward_message

    def predict_columns(self):
        self.prediction = np.dot(self.forward_message, self.transition_probs)
        prediction = np.reshape(self.prediction, (self.cells_per_column, self.n_columns))
        prediction = prediction.sum(axis=0)
        return prediction

    def reset(self):
        self.forward_message = np.ones(self.n_states) / self.n_states
        self.backward_message = np.ones(self.n_states) / self.n_states
        self.prediction = None

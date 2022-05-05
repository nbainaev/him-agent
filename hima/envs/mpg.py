#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np


class MarkovProcessGrammar:
    def __init__(
            self,
            n_states,
            transition_probs,
            transition_letters,
            alphabet,
            initial_state,
            terminal_state,
            autoreset=False,
            seed=None
    ):
        self.transition_probs = transition_probs
        self.transition_letters = transition_letters
        self.alphabet = alphabet
        self.char_to_num = {x: i for i, x in enumerate(alphabet)}

        self.states = np.arange(n_states)
        self.initial_state = initial_state
        self.terminal_state = terminal_state

        self.current_state = initial_state

        self.autoreset = autoreset

        self.rng = np.random.default_rng(seed)

    def set_current_state(self, state):
        self.current_state = state

    def reset(self):
        self.current_state = self.initial_state

    def next_state(self):
        if self.current_state == self.terminal_state:
            if self.autoreset:
                self.reset()
            else:
                return None

        transition_dist = self.transition_probs[self.current_state]

        new_state = self.rng.choice(self.states, p=transition_dist)

        letter = self.transition_letters[self.current_state][new_state]

        self.current_state = new_state

        return letter

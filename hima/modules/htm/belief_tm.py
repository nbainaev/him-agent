#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
from htm.advanced.support.numpy_helpers import setCompare, argmaxMulti, getAllCellsInColumns

EPS = 1e-24
UINT_DTYPE = "uint32"
REAL_DTYPE = "float32"
REAL64_DTYPE = "float64"
_TIE_BREAKER_FACTOR = 1e-24


class NaiveBayesTM:
    def __init__(
            self,
            n_columns,
            cells_per_column,
            max_segments_per_cell,
            max_receptive_field_size,
            w_lr,
            w_punish,
            theta_lr,
            b_lr,
            init_w=1,
            init_theta=1,
            init_b=0,
            seed=None
    ):
        # fixed parameters
        self.n_columns = n_columns
        self.cells_per_column = cells_per_column
        self.max_segments_per_cell = max_segments_per_cell

        if max_receptive_field_size == -1:
            self.max_receptive_field_size = n_columns * cells_per_column
        # learning rates
        self.w_lr = w_lr
        self.w_punish = w_punish
        self.theta_lr = theta_lr
        self.b_lr = b_lr

        # discrete states
        self.active_cells = np.empty(0, dtype=UINT_DTYPE)
        self.winner_cells = np.empty(0, dtype=UINT_DTYPE)
        self.predicted_cells = np.empty(0, dtype=UINT_DTYPE)
        self.predicted_columns = np.empty(0, dtype=UINT_DTYPE)
        self.predicted_segments = np.empty(0, dtype=UINT_DTYPE)
        self.active_columns = np.empty(0, dtype=UINT_DTYPE)

        # probabilities
        self.column_probs = np.zeros(
            self.n_columns, dtype=REAL64_DTYPE
        )
        self.cell_probs = np.zeros(
            self.n_columns * self.cells_per_column, dtype=REAL64_DTYPE
        )
        self.segment_probs = np.zeros(
            self.cell_probs.size * self.max_segments_per_cell, dtype=REAL64_DTYPE
        )

        # initial values
        self.init_w = init_w
        self.init_theta = init_theta
        self.init_b = init_b
        # learning parameters
        self.w = np.zeros((self.segment_probs.size, self.cell_probs.size), dtype=REAL64_DTYPE) + self.init_w
        self.receptive_fields = np.zeros(
            (self.segment_probs.size, self.cell_probs.size),
            dtype="bool"
        )
        self.theta = np.zeros(self.cell_probs.size, dtype=REAL64_DTYPE) + self.init_theta
        self.b = np.zeros(self.segment_probs.size, dtype=REAL64_DTYPE)

        # surprise (analog to anomaly in classical TM)
        self.surprise = 0
        self.anomaly = 0
        self.confidence = 0

        # init random generator
        self.rng = np.random.default_rng(seed)

    def set_active_columns(self, active_columns):
        self.active_columns = active_columns

    def reset(self):
        # discrete states
        self.active_cells = np.empty(0, dtype=UINT_DTYPE)
        self.winner_cells = np.empty(0, dtype=UINT_DTYPE)
        self.predicted_cells = np.empty(0, dtype=UINT_DTYPE)
        self.predicted_columns = np.empty(0, dtype=UINT_DTYPE)
        self.predicted_segments = np.empty(0, dtype=UINT_DTYPE)
        self.active_columns = np.empty(0, dtype=UINT_DTYPE)

        # probabilities
        self.column_probs = np.zeros(
            self.n_columns, dtype=REAL64_DTYPE
        )
        self.cell_probs = np.zeros(
            self.n_columns * self.cells_per_column, dtype=REAL64_DTYPE
        )
        self.segment_probs = np.zeros(
            self.cell_probs.size * self.max_segments_per_cell, dtype=REAL64_DTYPE
        )

    def activate_cells(self, learn=True):
        # compute surprise
        inactive_columns = np.flatnonzero(np.in1d(np.arange(self.n_columns), self.active_columns, invert=True))
        surprise = - np.sum(np.log(self.column_probs[self.active_columns]))
        surprise += - np.sum(np.log(1 - self.column_probs[inactive_columns]))
        self.surprise = surprise

        correct_predicted_cells, bursting_columns = setCompare(
            self.predicted_cells,
            self.active_columns,
            aKey=self._columns_for_cells(
                self.predicted_cells
            ),
            rightMinusLeft=True
        )

        self.anomaly = len(bursting_columns) / len(self.active_columns)
        self.confidence = len(self.predicted_columns) / len(self.active_columns)

        (true_positive_segments,
         false_positive_segments,
         new_active_segments,
         winner_cells_in_bursting_columns) = self._calculate_learning(
            bursting_columns,
            correct_predicted_cells
        )

        winner_cells = np.concatenate(
            [correct_predicted_cells,
                winner_cells_in_bursting_columns]
        )

        active_cells = np.concatenate(
            [correct_predicted_cells,
                getAllCellsInColumns(bursting_columns, self.cells_per_column)]
        )

        if learn:
            self._update_weights(
                true_positive_segments,
                false_positive_segments,
                new_active_segments,
                active_cells  # or winner cells?
            )

        # update active and winner cells
        self.active_cells = active_cells
        self.winner_cells = winner_cells

    def predict_cells(self):
        self.predicted_cells = np.unique(
            self._cells_for_segments(
                self.predicted_segments
            )
        )
        self.predicted_columns = np.unique(self._columns_for_cells(self.predicted_cells))

    def activate_dendrites(self, steps=1, use_probs=False):
        assert steps >= 1

        if use_probs:
            cell_probs = self.cell_probs
        else:
            cell_probs = np.zeros_like(self.cell_probs)
            cell_probs[self.active_cells] = 1

        segment_probs = np.zeros_like(self.b)
        segments_in_use = np.flatnonzero(np.sum(self.receptive_fields, axis=-1))
        w = self.w[segments_in_use]
        b = self.b[segments_in_use]
        f = self.receptive_fields[segments_in_use]
        for step in range(steps):
            likelihood = np.power((1-w), 1 - cell_probs)*np.power(w, cell_probs)

            z = ((self.theta / likelihood) - b.reshape((-1, 1))) / (1 - b).reshape((-1, 1))

            x = np.prod(z, axis=-1, where=f)

            segment_probs[segments_in_use] = np.clip(1 / (1 + np.nan_to_num((1 - b) * x, nan=np.inf, posinf=np.inf, neginf=-np.inf) / b), 0, 1)

            cell_probs = 1 - np.prod(1 - segment_probs.reshape(cell_probs.size, -1), axis=-1)

        self.predicted_segments = np.flatnonzero(
            np.random.random(segment_probs.size) < segment_probs
        )
        self.segment_probs = segment_probs
        self.cell_probs = cell_probs

        cell_probs = 1 - cell_probs.reshape((self.n_columns, -1))
        self.column_probs = 1 - np.prod(cell_probs, axis=-1)

    def _update_weights(
            self,
            true_positive_segments,
            false_positive_segments,
            new_active_segments,
            active_cells
    ):
        # update dendrites activity
        active_segments_dense = np.zeros_like(self.b)
        active_segments_dense[true_positive_segments] = 1
        active_segments_dense[new_active_segments] = 1

        b_deltas = active_segments_dense - self.b
        self.b += self.b_lr * b_deltas

        self.b = np.clip(self.b, 0, 1)

        # update cells' activity
        active_cells_dense = np.zeros_like(self.theta)
        active_cells_dense[active_cells] = 1

        theta_deltas = active_cells_dense - self.theta
        self.theta += self.theta_lr * theta_deltas

        self.theta = np.clip(self.theta, 0, 1)

        # update conditional probs
        old_winner_cells_dense = np.zeros_like(self.theta)
        old_winner_cells_dense[self.winner_cells] = 1

        if len(true_positive_segments) > 0:
            w_true_positive = self.w[true_positive_segments]
            w_deltas_tpos = old_winner_cells_dense - w_true_positive
            self.w[true_positive_segments] += self.w_lr * w_deltas_tpos
            self.receptive_fields[true_positive_segments] = self.receptive_fields[true_positive_segments] + old_winner_cells_dense

        w_false_positive = self.w[false_positive_segments]
        w_deltas_fpos = -old_winner_cells_dense * w_false_positive
        self.w[false_positive_segments] += self.w_punish * w_deltas_fpos

        self.w = np.clip(self.w, 0, 1)

        # init new segments
        if len(self.winner_cells) > 0:
            self.w[new_active_segments] = self.init_w
            self.b[new_active_segments] = self.init_b
            self.receptive_fields[new_active_segments] = old_winner_cells_dense

        # prune zero synapses
        self.receptive_fields[self.w == 0] = 0
        # reset zero segments
        zero_segments = np.flatnonzero(np.sum(self.receptive_fields, axis=-1) == 0)
        self.b[zero_segments] = 0

    def _calculate_learning(self, bursting_columns, correct_predicted_cells):
        true_positive_segments, false_positive_segments = setCompare(
            self.predicted_segments,
            correct_predicted_cells,
            aKey=self._cells_for_segments(self.predicted_segments),
            leftMinusRight=True
        )
        # choose cells with the least amount of segments in bursting columns
        cells_candidates = getAllCellsInColumns(
            bursting_columns,
            self.cells_per_column
        )
        tiebreaker = self.rng.random(cells_candidates.size)

        segments_in_use = np.flatnonzero(np.sum(self.receptive_fields, axis=-1))

        score = np.zeros_like(self.theta)
        if len(segments_in_use) > 0:
            cells, counts = np.unique(self._cells_for_segments(segments_in_use), return_counts=True)
            score[cells] = counts

        cells_scores = -score[cells_candidates] + tiebreaker * _TIE_BREAKER_FACTOR
        cells_to_grow_segment = cells_candidates[
            argmaxMulti(
                cells_scores,
                self._columns_for_cells(cells_candidates)
            )
        ]
        # choose the least used segment on cells
        segments_candidates = self._get_all_segments_in_cells(cells_to_grow_segment)
        tiebreaker = self.rng.random(segments_candidates.size)
        segments_scores = -self.b[segments_candidates] + tiebreaker * _TIE_BREAKER_FACTOR
        new_active_segments = segments_candidates[
            argmaxMulti(
                segments_scores,
                self._cells_for_segments(segments_candidates)
            )
        ]

        return (true_positive_segments.astype(UINT_DTYPE),
                false_positive_segments.astype(UINT_DTYPE),
                new_active_segments.astype(UINT_DTYPE),
                cells_to_grow_segment.astype(UINT_DTYPE))

    def _get_all_segments_in_cells(self, cells):
        return ((cells * self.max_segments_per_cell).reshape((-1, 1)) + np.arange(
            self.max_segments_per_cell, dtype="uint32"
        )).flatten()

    def _columns_for_cells(self, cells):
        """
        Calculates column numbers for cells
        :param cells: numpy array of cells id
        :return: numpy array of columns id for every cell
        """

        columns = cells // self.cells_per_column
        return columns.astype(UINT_DTYPE)

    def _cells_for_segments(self, segments):
        """
        Calculate cell numbers for segments
        :param segments: numpy array of segment id
        :return: numpy array of cell id for every segment
        """
        cells = segments // self.max_segments_per_cell
        return cells.astype(UINT_DTYPE)

    def _filter_segments_by_cell(self, segments, cells, invert=False):
        mask = np.isin(self._cells_for_segments(segments), cells, invert=invert)
        return segments[mask]

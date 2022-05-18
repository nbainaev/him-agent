#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
from htm.advanced.support.numpy_helpers import setCompare, argmaxMulti, getAllCellsInColumns
from hima.modules.htm.temporal_memory import GeneralFeedbackTM
from hima.modules.htm.connections import Connections
from htm.bindings.sdr import SDR

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
            max_receptive_field_size=-1,
            w_lr=0.01,
            nu_lr=0.01,
            b_lr=0.01,
            beta_lr=0.01,
            init_w=0.5,
            init_nu=0.5,
            init_b=0.5,
            init_beta=0.5,
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
        self.nu_lr = nu_lr
        self.b_lr = b_lr
        self.beta_lr = beta_lr

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
        self.init_nu = init_nu
        self.init_b = init_b
        self.init_beta = init_beta

        # learning parameters
        self.w = np.zeros(
            (self.segment_probs.size, self.cell_probs.size), dtype=REAL64_DTYPE
        ) + self.init_w
        self.nu = np.zeros(
            (self.segment_probs.size, self.cell_probs.size), dtype=REAL64_DTYPE
        ) + self.init_nu
        self.receptive_fields = np.zeros(
            (self.segment_probs.size, self.cell_probs.size),
            dtype="bool"
        )
        self.b = np.zeros(self.segment_probs.size, dtype=REAL64_DTYPE)

        self.beta = np.zeros(self.segment_probs.size, dtype=REAL64_DTYPE) + self.init_beta

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
        inactive_columns = np.flatnonzero(
            np.in1d(np.arange(self.n_columns), self.active_columns, invert=True)
        )
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
                new_active_segments,
                true_positive_segments,
                false_positive_segments,
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
        # non-zero segments' id
        segments_in_use = np.flatnonzero(np.sum(self.receptive_fields, axis=-1))
        if len(segments_in_use) > 0:
            w = self.w[segments_in_use]
            nu = self.nu[segments_in_use]
            b = self.b[segments_in_use]
            beta = self.beta[segments_in_use]
            f = self.receptive_fields[segments_in_use]
            cells_with_segments, indices = np.unique(
                self._cells_for_segments(segments_in_use),
                return_index=True
            )

            for step in range(steps):
                # p(s|d=1)
                synapse_probs_true = np.power((1 - w), 1 - cell_probs) * np.power(w, cell_probs)
                # p(s|d=0)
                synapse_probs_false = np.power((1 - nu), 1 - cell_probs) * np.power(nu, cell_probs)

                likelihood_true = np.prod(synapse_probs_true, axis=-1, where=f)
                likelihood_false = np.prod(synapse_probs_false, axis=-1, where=f)

                norm = b * likelihood_true + (1 - b) * likelihood_false
                segment_probs[segments_in_use] = np.divide(
                    b * likelihood_true, norm,
                    out=np.zeros_like(b, dtype=REAL64_DTYPE), where=(norm != 0)
                )

                not_active_prob = np.power((1 - beta), segment_probs[segments_in_use])
                not_active_prob = np.multiply.reduceat(not_active_prob, indices)

                cell_probs = np.zeros_like(self.cell_probs)
                cell_probs[cells_with_segments] = 1 - not_active_prob
        else:
            cell_probs = np.zeros_like(self.cell_probs)

        self.predicted_segments = np.flatnonzero(
            np.random.random(segment_probs.size) < segment_probs
        )
        self.segment_probs = segment_probs
        self.cell_probs = cell_probs

        cell_probs = 1 - cell_probs.reshape((self.n_columns, -1))
        self.column_probs = 1 - np.prod(cell_probs, axis=-1)

    def _update_weights(
            self,
            new_active_segments,
            true_positive_segments,
            false_positive_segments
    ):
        active_segments_dense = np.zeros_like(self.b)
        active_segments_dense[true_positive_segments] = 1

        old_winner_cells_dense = np.zeros_like(self.cell_probs)
        old_winner_cells_dense[self.winner_cells] = 1

        old_active_cells_dense = np.zeros_like(self.cell_probs)
        old_active_cells_dense[self.active_cells] = 1

        # init new segments
        if len(new_active_segments) > 0:
            self.w[new_active_segments] = self.init_w
            self.nu[new_active_segments] = self.init_nu
            self.b[new_active_segments] = self.init_b
            self.beta[new_active_segments] = self.init_beta
            self.receptive_fields[new_active_segments] = old_winner_cells_dense

        # update dendrites activity
        b_deltas = active_segments_dense - self.b
        self.b += self.b_lr * b_deltas

        self.b = np.clip(self.b, 0, 1)

        # update segment reliability
        if len(true_positive_segments) > 0:
            self.beta[true_positive_segments] += self.beta_lr * (
                        1 - self.beta[true_positive_segments])
        if len(false_positive_segments) > 0:
            self.beta[false_positive_segments] -= self.beta_lr * self.beta[false_positive_segments]

        self.beta = np.clip(self.beta, 0, 1)

        # update conditional probs
        if len(true_positive_segments) > 0:
            w_true_positive = self.w[true_positive_segments]
            w_deltas_tpos = old_active_cells_dense - w_true_positive
            w_deltas_tpos[~self.receptive_fields[true_positive_segments]] = 0

            self.w[true_positive_segments] += self.w_lr * w_deltas_tpos
            self.receptive_fields[true_positive_segments] = (self.receptive_fields[
                                                                 true_positive_segments] + old_winner_cells_dense).astype(
                'bool'
            )

        nu_deltas = old_active_cells_dense - self.nu
        nu_deltas[~self.receptive_fields] = 0
        nu_deltas[true_positive_segments] = 0
        self.nu += self.nu_lr * nu_deltas

        self.w = np.clip(self.w, 0, 1)
        self.nu = np.clip(self.nu, 0, 1)

        # prune zero synapses
        self.receptive_fields[self.w < EPS] = 0

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

        score = np.zeros_like(self.cell_probs)
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


class HybridNaiveBayesTM(GeneralFeedbackTM):
    def __init__(
            self,
            w_lr=0.01,
            nu_lr=0.01,
            b_lr=0.01,
            beta_lr=0.01,
            theta_lr=0.01,
            tau_lr=0.01,
            gamma_lr=0.01,
            init_w=0.5,
            init_nu=0.5,
            init_b=0.5,
            init_beta=0.5,
            init_theta=0.5,
            init_tau=0.5,
            init_gamma=0.5,
            full_learning=False,
            max_interneurons=100,
            connected_threshold_inhib=0.01,
            learning_threshold_inhib=1,
            initial_permanence_inhib=0.5,
            permanence_increment_inhib=0.1,
            permanence_decrement_inhib=0.1,
            **kwargs
    ):
        super(HybridNaiveBayesTM, self).__init__(**kwargs)

        # learning rates
        self.w_lr = w_lr
        self.nu_lr = nu_lr
        self.b_lr = b_lr
        self.beta_lr = beta_lr
        self.theta_lr = theta_lr
        self.tau_lr = tau_lr
        self.gamma_lr = gamma_lr

        self.segments_in_use = np.empty(0, dtype=UINT_DTYPE)
        # probabilities
        self.column_probs = np.zeros(
            self.columns, dtype=REAL64_DTYPE
        )
        self.cell_probs = np.zeros(
            self.columns * self.cells_per_column, dtype=REAL64_DTYPE
        )
        self.segment_probs = np.zeros(
            self.cell_probs.size * self.max_segments_per_cell_basal, dtype=REAL64_DTYPE
        )

        # initial values
        self.init_w = init_w
        self.init_nu = init_nu
        self.init_b = init_b
        self.init_beta = init_beta
        self.init_theta = init_theta
        self.init_gamma = init_gamma
        self.init_tau = init_tau

        self.full_learning = full_learning
        # learning parameters
        self.w = np.zeros(
            (self.segment_probs.size, self.cell_probs.size), dtype=REAL64_DTYPE
        ) + self.init_w
        self.nu = np.zeros(
            (self.segment_probs.size, self.cell_probs.size), dtype=REAL64_DTYPE
        ) + self.init_nu
        self.receptive_fields = np.zeros(
            (self.segment_probs.size, self.cell_probs.size),
            dtype="bool"
        )
        self.b = np.zeros(self.segment_probs.size, dtype=REAL64_DTYPE)

        self.beta = np.zeros(self.segment_probs.size, dtype=REAL64_DTYPE) + self.init_beta

        # inhibitory interneurons
        self.max_interneurons = max_interneurons
        self.connected_threshold_inhib = connected_threshold_inhib
        self.learning_threshold_inhib = learning_threshold_inhib
        self.initial_permanence_inhib = initial_permanence_inhib
        self.presynaptic_inhib_cells = SDR(self.local_cells + self.max_interneurons)
        self.permanence_increment_inhib = permanence_increment_inhib
        self.permanence_decrement_inhib = permanence_decrement_inhib

        self.inhib_connections = Connections(numCells=self.local_cells + self.max_interneurons,
                                             connectedThreshold=self.connected_threshold_inhib,
                                             timeseries=self.timeseries)

        self.theta = np.zeros(
            (self.max_interneurons, self.cell_probs.size), dtype=REAL64_DTYPE
        ) + self.init_theta
        self.tau = np.zeros(
            (self.max_interneurons, self.cell_probs.size), dtype=REAL64_DTYPE
        ) + self.init_tau
        self.inhib_receptive_fields = np.zeros(
            (self.max_interneurons, self.cell_probs.size),
            dtype="bool"
        )

        self.gamma = np.zeros(
            self.max_interneurons, dtype=REAL64_DTYPE
        ) + self.init_gamma

        # surprise (analog to anomaly in classical TM)
        self.surprise = 0

        self.np_rng = np.random.default_rng(kwargs['seed'])

    def reset(self):
        super(HybridNaiveBayesTM, self).reset()

        # probabilities
        self.column_probs = np.zeros(
            self.columns, dtype=REAL64_DTYPE
        )
        self.cell_probs = np.zeros(
            self.columns * self.cells_per_column, dtype=REAL64_DTYPE
        )
        self.segment_probs = np.zeros(
            self.cell_probs.size * self.max_segments_per_cell_basal, dtype=REAL64_DTYPE
        )

    def activate_cells(self, learn: bool):
        """
                Calculates new active cells and performs connections' learning.
                :param learn: if true, connections will learn patterns from previous step
                :return:
                """
        # compute surprise
        inactive_columns = np.flatnonzero(
            np.in1d(np.arange(self.columns), self.get_active_columns(), invert=True)
        )
        surprise = - np.sum(np.log(self.column_probs[self.get_active_columns()]))
        surprise += - np.sum(np.log(1 - self.column_probs[inactive_columns]))
        self.surprise = surprise

        # Calculate active cells
        correct_predicted_cells, bursting_columns = setCompare(
            self.predicted_cells.sparse, self.active_columns.sparse,
            aKey=self._columns_for_cells(
                self.predicted_cells.sparse
            ),
            rightMinusLeft=True
        )
        self.correct_predicted_cells.sparse = correct_predicted_cells
        new_active_cells = np.concatenate(
            (correct_predicted_cells,
             getAllCellsInColumns(
                bursting_columns,
                self.cells_per_column
             ) + self.local_range[0])
        )

        (learning_active_basal_segments,
         learning_matching_basal_segments,
         learning_matching_apical_segments,
         cells_to_grow_apical_segments,
         basal_segments_to_punish,
         apical_segments_to_punish,
         cells_to_grow_apical_and_basal_segments,
         new_winner_cells) = self._calculate_learning(bursting_columns, correct_predicted_cells)

        # Learn
        if learn:
            if len(new_winner_cells) > 0:
                self._learn_inhib_cell(new_winner_cells)
            # Learn on existing segments
            if self.active_cells_context.sparse.size > 0:
                for learning_segments in (
                        learning_active_basal_segments, learning_matching_basal_segments):
                    self._learn(
                        self.basal_connections, learning_segments, self.active_cells_context,
                        self.active_cells_context.sparse,
                        self.num_potential_basal, self.sample_size_basal,
                        self.max_synapses_per_segment_basal,
                        self.initial_permanence_basal, self.permanence_increment_basal,
                        self.permanence_decrement_basal,
                        self.learning_threshold_basal
                    )
            if self.active_cells_feedback.sparse.size > 0:
                self._learn(
                    self.apical_connections, learning_matching_apical_segments,
                    self.active_cells_feedback,
                    self.active_cells_feedback.sparse,
                    self.num_potential_apical, self.sample_size_apical,
                    self.max_synapses_per_segment_apical,
                    self.initial_permanence_apical, self.permanence_increment_apical,
                    self.permanence_decrement_apical,
                    self.learning_threshold_apical
                )

            # Punish incorrect predictions
            if self.predicted_segment_decrement_basal != 0.0:
                if self.active_cells_context.sparse.size > 0:
                    for segment in basal_segments_to_punish:
                        self.basal_connections.adaptSegment(
                            segment, self.active_cells_context,
                            -self.predicted_segment_decrement_basal, 0.0,
                            self.prune_zero_synapses, self.learning_threshold_basal
                        )
                if self.active_cells_feedback.sparse.size > 0:
                    for segment in apical_segments_to_punish:
                        self.apical_connections.adaptSegment(
                            segment, self.active_cells_feedback,
                            -self.predicted_segment_decrement_apical, 0.0,
                            self.prune_zero_synapses, self.learning_threshold_apical
                        )

            # Grow new segments
            if self.active_cells_context.sparse.size > 0:
                new_basal_segments = self._learn_on_new_segments(
                    self.basal_connections,
                    cells_to_grow_apical_and_basal_segments,
                    self.active_cells_context.sparse,
                    self.sample_size_basal, self.max_synapses_per_segment_basal,
                    self.initial_permanence_basal,
                    self.max_segments_per_cell_basal
                )
            else:
                new_basal_segments = np.empty(0)

            if self.active_cells_feedback.sparse.size > 0:
                new_apical_segments = self._learn_on_new_segments(
                    self.apical_connections,
                    np.concatenate(
                        (cells_to_grow_apical_segments,
                         cells_to_grow_apical_and_basal_segments)
                    ),
                    self.active_cells_feedback.sparse,
                    self.sample_size_apical, self.max_synapses_per_segment_apical,
                    self.initial_permanence_apical,
                    self.max_segments_per_cell_apical
                )
            else:
                new_apical_segments = np.empty(0)

            self._update_receptive_fields()
            self._update_weights(
                new_winner_cells,
                new_basal_segments,
                learning_active_basal_segments,
                basal_segments_to_punish
            )

        self.active_cells.sparse = np.unique(new_active_cells.astype('uint32'))
        self.winner_cells.sparse = np.unique(new_winner_cells)

        n_active_columns = self.active_columns.sparse.size
        self.mean_active_columns = self.sm_ac * self.mean_active_columns + (
                1 - self.sm_ac) * n_active_columns
        if n_active_columns != 0:
            anomaly = len(bursting_columns) / n_active_columns
        else:
            anomaly = 1.0

        self.anomaly_threshold = self.anomaly_threshold + (
                anomaly - self.anomaly[0]) / self.anomaly_window
        self.anomaly.append(anomaly)
        self.anomaly.pop(0)

    def predict_columns(self, steps=1, use_probs=False, update_segments=True):
        assert steps >= 1

        if use_probs:
            cell_probs = self.cell_probs
        else:
            cell_probs = np.zeros_like(self.cell_probs)
            cell_probs[self.get_active_cells_context()] = 1

        if update_segments:
            self._update_segments_in_use(sort=True)

        segment_probs = np.zeros_like(self.segments_in_use)

        if len(self.segments_in_use) > 0:
            cells_for_segments = self.basal_connections.mapSegmentsToCells(self.segments_in_use) - \
                                 self.local_range[0]

            cells_with_segments, indices = np.unique(cells_for_segments, return_index=True)

            w = self.w[self.segments_in_use]
            nu = self.nu[self.segments_in_use]
            b = self.b[self.segments_in_use]
            beta = self.beta[self.segments_in_use]
            f = self.receptive_fields[self.segments_in_use]

            for step in range(steps):
                # p(s|d=1)
                synapse_probs_true = np.power((1 - w), 1 - cell_probs) * np.power(w, cell_probs)
                # p(s|d=0)
                synapse_probs_false = np.power((1 - nu), 1 - cell_probs) * np.power(nu, cell_probs)

                likelihood_true = np.prod(synapse_probs_true, axis=-1, where=f)
                likelihood_false = np.prod(synapse_probs_false, axis=-1, where=f)

                norm = b * likelihood_true + (1 - b) * likelihood_false
                segment_probs = np.divide(
                    b * likelihood_true, norm,
                    out=np.zeros_like(b, dtype=REAL64_DTYPE), where=(norm != 0)
                )

                not_active_prob = np.multiply.reduceat(np.power((1 - beta), segment_probs), indices)

                cell_probs = np.zeros_like(self.cell_probs)
                cell_probs[cells_with_segments] = 1 - not_active_prob
        else:
            cell_probs = np.zeros_like(self.cell_probs)

        self.segment_probs = np.zeros_like(self.b)
        if len(self.segments_in_use) > 0:
            self.segment_probs[self.segments_in_use] = segment_probs

        self.cell_probs = cell_probs

        cell_probs = 1 - cell_probs.reshape((self.columns, -1))
        self.column_probs = 1 - np.prod(cell_probs, axis=-1)

    def sample_cells(self):
        cell_probs = self.cell_probs
        # non-zero segments' id
        segments_in_use = np.flatnonzero(np.sum(self.inhib_receptive_fields, axis=-1))
        theta = self.theta[segments_in_use]
        tau = self.tau[segments_in_use]
        gamma = self.gamma[segments_in_use]
        f = self.inhib_receptive_fields[segments_in_use]

        # p(s|d=1)
        synapse_probs_true = np.power((1 - theta), 1 - cell_probs) * np.power(theta, cell_probs)
        # p(s|d=0)
        synapse_probs_false = np.power((1 - tau), 1 - cell_probs) * np.power(tau, cell_probs)

        likelihood_true = np.prod(synapse_probs_true, axis=-1, where=f)
        likelihood_false = np.prod(synapse_probs_false, axis=-1, where=f)

        norm = gamma * likelihood_true + (1 - gamma) * likelihood_false
        segment_probs = np.divide(
            gamma * likelihood_true, norm,
            out=np.zeros_like(gamma, dtype=REAL64_DTYPE), where=(norm != 0)
        )

        norm2 = np.sum(theta, axis=-1, where=f)
        cluster_probs = np.sum(theta * self.cell_probs, axis=-1, where=f)
        cluster_probs = np.divide(
            cluster_probs,
            norm2,
            out=np.zeros_like(cluster_probs, dtype=REAL64_DTYPE), where=(norm2 != 0)
        )

        cluster_probs *= segment_probs

        # normalize
        norm3 = cluster_probs.sum()
        if norm3 != 0:
            cluster_probs /= norm3
            # choose cluster
            segment = self.np_rng.choice(segments_in_use, 1, p=cluster_probs)

            # sample cells from cluster
            cell_probs = self.theta[segment] * self.inhib_receptive_fields[segment]
            cells = np.flatnonzero(self.np_rng.random(cell_probs.size) < cell_probs)
        else:
            segment = np.empty(0, dtype=UINT_DTYPE)
            cells = np.empty(0, dtype=UINT_DTYPE)

        return cells

    def _learn_inhib_cell(self, winner_cells):
        self.presynaptic_inhib_cells.sparse = winner_cells

        num_connected, num_potential = self.inhib_connections.computeActivityFull(
            self.presynaptic_inhib_cells,
            True
        )
        # TODO should we use num_connected instead?
        candidates = np.flatnonzero(num_potential >= self.learning_threshold_inhib)
        if len(candidates) > 0:
            winner = candidates[np.argmax(num_potential[candidates])]
            is_new = False
        else:
            winner = np.argmin(self.gamma)
            is_new = True

        winner_cells_dense = np.zeros(self.local_cells)
        winner_cells_dense[winner_cells] = 1

        if is_new:
            new_segment = self.inhib_connections.createSegment(winner, 1)
            self.inhib_connections.growSynapses(
                new_segment, winner_cells, self.initial_permanence_inhib, self.rng,
                maxNew=len(winner_cells)
            )

            self.theta[new_segment] = self.init_theta
            self.tau[new_segment] = self.init_tau
            self.gamma[new_segment] = self.init_gamma

            self.inhib_receptive_fields[new_segment] = winner_cells_dense
        else:
            segment = self.inhib_connections.getSegment(winner, 0)
            self.inhib_connections.adaptSegment(
                segment, self.presynaptic_inhib_cells, self.permanence_increment_inhib, self.permanence_decrement_inhib,
                self.prune_zero_synapses, self.learning_threshold_inhib
            )
            max_new = len(winner_cells) - num_potential[winner]
            if max_new > 0:
                self.inhib_connections.growSynapses(segment, winner_cells, self.initial_permanence_inhib, self.rng, max_new)

            cells = np.array(self.inhib_connections.presynapticCellsForSegment(segment))
            cells_dense = np.zeros_like(self.cell_probs, dtype='bool')
            cells_dense[cells] = 1
            self.inhib_receptive_fields[segment] = cells_dense

            # update dendrites activity
            active_segments_dense = np.zeros_like(self.gamma)
            active_segments_dense[segment] = 1

            gamma_deltas = active_segments_dense - self.gamma
            self.gamma += self.gamma_lr * gamma_deltas

            self.gamma = np.clip(self.gamma, 0, 1)

            # update conditional probs
            theta_active = self.theta[segment]
            theta_delta = winner_cells_dense - theta_active
            theta_delta[~self.inhib_receptive_fields[segment]] = 0

            self.theta[segment] += self.theta_lr * theta_delta

            tau_deltas = winner_cells_dense - self.tau
            tau_deltas[~self.inhib_receptive_fields] = 0
            tau_deltas[segment] = 0
            self.tau += self.tau_lr * tau_deltas

            self.theta = np.clip(self.theta, 0, 1)
            self.tau = np.clip(self.tau, 0, 1)

    def _update_segments_in_use(self, sort=True):
        # non-zero segments' id
        segments_in_use = np.flatnonzero(np.sum(self.receptive_fields, axis=-1))

        if sort and (len(segments_in_use) > 0):
            cells_for_segments = self.basal_connections.mapSegmentsToCells(segments_in_use) - \
                                 self.local_range[0]

            sorter = np.argsort(cells_for_segments)

            segments_in_use = segments_in_use[sorter]

            self.segments_in_use = segments_in_use

    def _learn_on_new_segments(
            self, connections: Connections, new_segment_cells, growth_candidates, sample_size,
            max_synapses_per_segment,
            initial_permanence, max_segments_per_cell
    ):
        """
        Grows new segments and learn on them
        :param connections:
        :param new_segment_cells: cells' id to grow new segments on
        :param growth_candidates: cells' id to grow synapses to
        :return:
        """
        num_new_synapses = len(growth_candidates)

        if sample_size != -1:
            num_new_synapses = min(num_new_synapses, sample_size)

        if max_synapses_per_segment != -1:
            num_new_synapses = min(num_new_synapses, max_synapses_per_segment)

        new_segments = list()
        for cell in new_segment_cells:
            new_segment = connections.createSegment(cell, max_segments_per_cell)
            new_segments.append(new_segment)
            connections.growSynapses(
                new_segment, growth_candidates, initial_permanence, self.rng,
                maxNew=num_new_synapses
            )

        return np.array(new_segments, dtype=UINT_DTYPE)

    def _update_weights(
            self,
            new_winner_cells,
            new_active_segments,
            true_positive_segments,
            false_positive_segments
    ):
        # non-zero segments' id
        segments_in_use = np.flatnonzero(np.sum(self.receptive_fields, axis=-1))
        # TODO should it be only matched and active segments?
        active_segments = self.basal_connections.filterSegmentsByCell(segments_in_use, new_winner_cells)

        # init new segments
        if len(new_active_segments) > 0:
            self.w[new_active_segments] = self.init_w
            self.nu[new_active_segments] = self.init_nu
            self.b[new_active_segments] = self.init_b
            self.beta[new_active_segments] = self.init_beta
        # update dendrites activity
        active_segments_dense = np.zeros_like(self.b)
        active_segments_dense[active_segments] = 1

        b_deltas = active_segments_dense - self.b
        self.b += self.b_lr * b_deltas

        self.b = np.clip(self.b, 0, 1)

        # update segment reliability
        if len(true_positive_segments) > 0:
            self.beta[true_positive_segments] += self.beta_lr * (1 - self.beta[true_positive_segments])
        if len(false_positive_segments) > 0:
            self.beta[false_positive_segments] -= self.beta_lr * self.beta[false_positive_segments]

        self.beta = np.clip(self.beta, 0, 1)

        # update conditional probs
        old_active_cells_dense = np.zeros_like(self.cell_probs)
        old_active_cells_dense[self.get_active_cells()] = 1

        if len(active_segments) > 0:
            w_active = self.w[active_segments]
            w_deltas = old_active_cells_dense - w_active
            w_deltas[~self.receptive_fields[active_segments]] = 0

            self.w[active_segments] += self.w_lr * w_deltas

        if self.full_learning:
            nu_deltas = old_active_cells_dense - self.nu
            nu_deltas[~self.receptive_fields] = 0
            nu_deltas[active_segments] = 0
            self.nu += self.nu_lr * nu_deltas
        else:
            if len(false_positive_segments) > 0:
                nu_false_positive = self.nu[false_positive_segments]
                nu_deltas = old_active_cells_dense - nu_false_positive
                nu_deltas[~self.receptive_fields[false_positive_segments]] = 0

                self.nu[false_positive_segments] += self.nu_lr * nu_deltas

        self.w = np.clip(self.w, 0, 1)
        self.nu = np.clip(self.nu, 0, 1)

    def _update_receptive_fields(self, segments=None):
        if segments is None:
            self.receptive_fields = np.zeros(
                (self.segment_probs.size, self.cell_probs.size),
                dtype="bool"
            )
            for cell in range(*self.local_range):
                cell_segments = self.basal_connections.segmentsForCell(cell)
                for segment in cell_segments:
                    cells = np.array(self.basal_connections.presynapticCellsForSegment(segment)) - \
                            self.context_range[0]
                    cells_dense = np.zeros_like(self.cell_probs, dtype='bool')
                    cells_dense[cells] = 1
                    self.receptive_fields[segment] = cells_dense
        else:
            for segment in segments:
                cells = np.array(self.basal_connections.presynapticCellsForSegment(segment)) - self.context_range[0]
                cells_dense = np.zeros_like(self.cell_probs, dtype='bool')
                cells_dense[cells] = 1
                self.receptive_fields[segment] = cells_dense

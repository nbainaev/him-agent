#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
import pickle
from hima.experiments.temporal_pooling.ablation_utp import AblationUtp
from hima.experiments.temporal_pooling.config import *
from hima.experiments.temporal_pooling.custom_utp import CustomUtp
from hima.experiments.temporal_pooling.data_generation import generate_data
from hima.experiments.temporal_pooling.metrics import (
    symmetric_error, representations_intersection_1, row_similarity,
    representation_similarity, similarity_mae
)
from hima.experiments.temporal_pooling.sandwich_tp import SandwichTp
from hima.experiments.temporal_pooling.utils import StupidEncoder, IdentityEncoder
from hima.modules.htm.spatial_pooler import UnionTemporalPooler
from hima.modules.htm.temporal_memory import DelayedFeedbackTM, ClassicTemporalMemory
from htm.algorithms import TemporalMemory
from htm.bindings.sdr import SDR


def train_model(tm: TemporalMemory, sdrs: np.ndarray, num_epochs=10) -> list:
    errors = []
    for epoch in range(num_epochs):
        for sdr in sdrs:
            tm.compute(sdr, learn=True)
            tm.activateDendrites(True)
            errors.append(tm.anomaly)
        tm.compute(SDR(sdrs[0].dense.shape), learn=False)
    return errors


def run(tm, tp, policy, state_encoder, action_encoder, learn=True, prev_dense=None, whole_active: SDR = None):
    tp_prev_union = tp.getUnionSDR().sparse.copy()
    tp_input = SDR(tp.getNumInputs())
    tp_predictive = SDR(tp.getNumInputs())
    window_size = 1
    window_error = 0
    counter = 0
    for state, action in policy:
        context = state_encoder.encode(state)
        active_input = action_encoder.encode(action)

        tm.set_active_context_cells(context)

        tm.activate_basal_dendrites(learn)

        tm.set_active_feedback_cells(tp.getUnionSDR().sparse)
        tm.activate_apical_dendrites(learn)
        tm.propagate_feedback()

        tm.predict_cells()

        tm.set_active_columns(active_input)
        tm.activate_cells(learn)

        tp_input.sparse = tm.get_active_cells()
        tp_predictive.sparse = tm.get_correctly_predicted_cells()
        tp.compute(tp_input, tp_predictive, learn)

        current_union = tp.getUnionSDR().sparse.copy()

        window_error += symmetric_error(current_union, tp_prev_union)

        my_log = {}
        if prev_dense is not None:
            my_log['new_cells_percent'] = 1 - representations_intersection_1(tp.getUnionSDR().dense, prev_dense)
            my_log['prev_similarity'] = representation_similarity(tp.getUnionSDR().dense, prev_dense)
            my_log['num_in_prev'] = np.count_nonzero(prev_dense)

        my_log['num_in_curr'] = np.count_nonzero(tp.getUnionSDR().dense)

        if whole_active is not None:
            whole_active.dense = np.logical_or(whole_active.dense, tp.getUnionSDR().dense)
            whole_nonzero = np.count_nonzero(whole_active.dense)
            my_log['cells_in_whole'] = np.count_nonzero(tp.getUnionSDR().dense) / whole_nonzero

        if counter % window_size == window_size - 1:
            my_log['difference'] = (window_error / window_size)
            try:
                my_log['nonzero_pooling'] = np.count_nonzero(tp._pooling_activations)
                my_log['lower_bound'] = np.partition(
                    tp._pooling_activations.flatten(), -tp.cells_in_union-1
                )[-tp.cells_in_union-1]
            except BaseException:
                pass
            window_error = 0
        wandb.log(my_log)
        tp_prev_union = current_union.copy()

        counter += 1


def run_only_tp(_tp, sequence, _encoder, learn=True):
    prev_union = _tp.getUnionSDR().sparse.copy()

    for element in sequence:
        _input = _encoder.encode(element)
        sdr = SDR(_tp.getNumInputs())
        sdr.dense = _input
        _tp.compute(sdr, sdr, True)
        diff = symmetric_error(prev_union, _tp.getUnionSDR().sparse)
        wandb.log({'difference': diff})


def train_all_seq(tm, tp, data, state_encoder, action_encoder, iters_per_seq):
    representations = []
    prev = tp.getUnionSDR().dense.copy()
    for policy in data:
        tp.reset()
        whole_active = np.zeros(tp.getUnionSDR().dense.shape)
        for i in range(iters_per_seq):

            if i < 2:
                whole_active = None
            elif i == 2:
                whole_active = SDR(tp.getUnionSDR().dense.shape)
                whole_active.dense = np.zeros(tp.getUnionSDR().dense.shape)

            run(tm, tp, policy, state_encoder, action_encoder, learn=True, prev_dense=prev, whole_active=whole_active)

        representations.append(tp.getUnionSDR())
        prev = tp.getUnionSDR().dense.copy()
    return representations


def common_utp_one_seq(data):
    wandb.login()
    np.random.shuffle(data)

    wandb.init(project=wandb_project, entity=wandb_entity, reinit=True)
    tm = DelayedFeedbackTM(**config_tm)
    tp = UnionTemporalPooler(**config_tp)

    for i in range(100):
        run(tm, tp, data[0], state_encoder, action_encoder, True)
    wandb.finish()


def custom_utp_one_seq(data):
    wandb.init(project=wandb_project, entity=wandb_entity, reinit=True, config=utp_conf)

    # -----------------------------
    my_utp = CustomUtp(**utp_conf)
    tm = DelayedFeedbackTM(**config_tm)

    for i in range(100):
        run(tm, my_utp, data[0], state_encoder, action_encoder, True)

    wandb.finish(quiet=True)

    print(my_utp.getUnionSDR().dense.nonzero()[0].size / output_columns)


def only_custom_utp_test(row_data):
    wandb.init(project=wandb_project, entity=wandb_entity, reinit=True, config=utp_conf)

    my_utp = CustomUtp(**utp_conf)

    for i in range(100):
        run_only_tp(my_utp, row_data[0], StupidEncoder(n_actions, utp_conf['inputDimensions'][0]), True)

    wandb.finish(quiet=True)


def custom_utp_all_seq_5_epochs(data):
    my_utp = CustomUtp(**utp_conf)
    tm = DelayedFeedbackTM(**config_tm)
    all_seq(tm, my_utp, data, epochs=5)


def common_utp_all_seq_5_epochs(data):
    tp = UnionTemporalPooler(**config_tp)
    tm = DelayedFeedbackTM(**config_tm)
    all_seq(tm, tp, data, epochs=5)


def no_boosting(data):
    tp = AblationUtp(
        **config_tp,
        first_boosting=False,
        second_boosting=False
    )
    tm = DelayedFeedbackTM(**config_tm)
    all_seq(tm, tp, data, epochs=5)


def no_second_boosting(data):
    tp = AblationUtp(
        **config_tp,
        second_boosting=False
    )
    tm = DelayedFeedbackTM(**config_tm)
    all_seq(tm, tp, data, epochs=5)


def no_history_learning_5_epochs(data):
    tp = AblationUtp(
        **config_tp,
        history_learning=False
    )
    tm = DelayedFeedbackTM(**config_tm)
    all_seq(tm, tp, data, epochs=5)


def no_history_learning_15_epochs(data):
    tp = AblationUtp(
        **config_tp,
        history_learning=False
    )
    tm = DelayedFeedbackTM(**config_tm)
    all_seq(tm, tp, data, epochs=15)


def no_untemporal_learning(data):
    tp = AblationUtp(
        **config_tp,
        untemporal_learning=False
    )
    tm = DelayedFeedbackTM(**config_tm)
    all_seq(tm, tp, data, epochs=5)


def no_union_learning(data):
    tp = AblationUtp(
        **config_tp,
        union_learning=False
    )
    tm = DelayedFeedbackTM(**config_tm)
    all_seq(tm, tp, data, epochs=5)


def common_only_UnionL(data):
    tp = AblationUtp(
        **config_tp,
        untemporal_learning=False,
        union_learning=True,
        history_learning=False,
    )
    tm = DelayedFeedbackTM(**config_tm)
    all_seq(tm, tp, data, epochs=5)


def all_seq(tm, tp, data, epochs):
    wandb.init(project=wandb_project, entity=wandb_entity, reinit=True, config=utp_conf)

    representations = []

    for epoch in range(epochs):
        representations = train_all_seq(tm, tp, data, state_encoder, action_encoder, 20)

    vis_what(data, representations)

    wandb.finish(quiet=True)


def vis_what(data, representations):
    similarity_matrix = np.zeros((len(representations), len(representations)))
    pure_similarity = np.zeros(similarity_matrix.shape)
    for i, policy1 in enumerate(data):
        for j, policy2 in enumerate(data):
            pure_similarity[i][j] = row_similarity(policy1, policy2)
            similarity_matrix[i][j] = abs(
                representation_similarity(representations[i].dense, representations[j].dense)
            )

    fig = plt.figure(figsize=(40, 10))
    ax1 = fig.add_subplot(131)
    ax1.set_title('representational', size=40)
    ax2 = fig.add_subplot(132)
    ax2.set_title('pure', size=40)
    ax3 = fig.add_subplot(133)
    ax3.set_title('difference', size=40)

    sns.heatmap(similarity_matrix, vmin=0, vmax=1, cmap='plasma', ax=ax1)
    sns.heatmap(pure_similarity, vmin=0, vmax=1, cmap='plasma', ax=ax2)

    sns.heatmap(abs(pure_similarity - similarity_matrix), vmin=0, vmax=1, cmap='plasma', ax=ax3, annot=True)
    wandb.log({'representations similarity': wandb.Image(ax1)})
    wandb.run.summary['mae'] = similarity_mae(pure_similarity, similarity_matrix)
    plt.show()


def stp_all_seq_3_epochs(data):

    tm = DelayedFeedbackTM(**config_tm)
    stp = SandwichTp(**stp_config)

    all_seq(tm, stp, data, epochs=3)


# ----------------------------------------------------------------


def custom_test(data):
    tp = CustomUtp(**utp_conf)
    tm = DelayedFeedbackTM(**config_tm)
    all_seq(tm, tp, data, epochs=5)


def custom_only_union_l_uncut(data):
    tp = CustomUtp(
        **utp_conf,
        untemporal_learning_enabled=False,
        union_learning_enabled=True,
        history_learning_enabled=False,
        limit_union_cells=False
    )
    tm = DelayedFeedbackTM(**config_tm)
    all_seq(tm, tp, data, epochs=5)


def custom_no_history_l_uncut(data):
    tp = CustomUtp(**utp_conf, history_learning_enabled=False, limit_union_cells=False)
    tm = DelayedFeedbackTM(**config_tm)
    all_seq(tm, tp, data, epochs=5)


def custom_no_history_l(data):
    tp = CustomUtp(**utp_conf, history_learning_enabled=False)
    tm = DelayedFeedbackTM(**config_tm)
    all_seq(tm, tp, data, epochs=5)

# ---------------------------------------------------------------


def run_states_seq(tm: ClassicTemporalMemory, tp, states, state_encoder, learn=True, prev_dense=None, whole_active: SDR=None):
    tp_prev_union = tp.getUnionSDR().sparse.copy()
    tp_input = SDR(tp.getNumInputs())
    tp_predictive = SDR(tp.getNumInputs())
    window_size = 1
    window_error = 0
    counter = 0
    for state in states:
        active_input = state_encoder.encode(state)

        tm.compute(active_input, learn=learn)
        tm.activateDendrites(learn=learn)

        tp_input.sparse = tm.getActiveCells().sparse.copy()
        tp_predictive.sparse = np.intersect1d(tm.getPredictiveCells().sparse.copy(), tp_input.sparse)
        tp.compute(tp_input, tp_predictive, learn)

        current_union = tp.getUnionSDR().sparse.copy()

        window_error += symmetric_error(current_union, tp_prev_union)

        my_log = {}
        if prev_dense is not None:
            my_log['new_cells_percent'] = 1 - representations_intersection_1(tp.getUnionSDR().dense, prev_dense)
            my_log['prev_similarity'] = representation_similarity(tp.getUnionSDR().dense, prev_dense)
            my_log['num_in_prev'] = np.count_nonzero(prev_dense)

        my_log['num_in_curr'] = np.count_nonzero(tp.getUnionSDR().dense)

        if whole_active is not None:
            whole_active.dense = np.logical_or(whole_active.dense, tp.getUnionSDR().dense)
            whole_nonzero = np.count_nonzero(whole_active.dense)
            my_log['cells_in_whole'] = np.count_nonzero(tp.getUnionSDR().dense) / whole_nonzero

        if counter % window_size == window_size - 1:
            my_log['difference'] = (window_error / window_size)
            try:
                my_log['nonzero_pooling'] = np.count_nonzero(tp._pooling_activations)
                my_log['lower_bound'] = np.partition(
                    tp._pooling_activations.flatten(), -tp.cells_in_union-1
                )[-tp.cells_in_union-1]
            except BaseException:
                pass

            try:
                my_log['tm.anomaly'] = tm.anomaly
            except BaseException:
                pass
            window_error = 0
        wandb.log(my_log)
        tp_prev_union = current_union.copy()

        counter += 1


def states_seqs_test(dataset):
    print('Init tm ==>')
    tm = ClassicTemporalMemory(**config_tm_classic)
    print('<== Init tm')
    print('Init tp ==>')
    tp = AblationUtp(**config_tp_for_classic_tm)
    print('<== Init tp ')
    state_encoder_ = IdentityEncoder()

    wandb.init(project=wandb_project, entity=wandb_entity, reinit=True, config=config_tm_classic)
    representations = []
    for states in dataset:
        tp.reset()
        for i in range(20):
            run_states_seq(tm, tp, states, state_encoder_)
        representations.append(tp.getUnionSDR())

    wandb.run.summary['rooms similarity'] = representation_similarity(representations[0].dense, representations[1].dense)
    wandb.finish()


def get_sdr_from_sparse(sparse_data, shape):
    if type(sparse_data[0]) == list:
        datasets = []
        for data in sparse_data:
            datasets.append(get_sdr_from_sparse(data, shape))
        return datasets

    sdr = []
    for data in sparse_data:
        sdr_ = SDR(shape)
        sdr_.sparse = data
        sdr.append(sdr_)
    return sdr


def _run_tests():
    wandb.login()
    np.random.seed(42)

    row_data, data = generate_data(5, n_actions, n_states, randomness=0.7)
    np.random.shuffle(data)
    # common_utp_one_seq(data)
    # custom_utp_one_seq(data)
    # only_custom_utp_test(row_data)
    # custom_utp_all_seq_5_epochs(data)
    # stp_all_seq_3_epochs(data)
    # common_utp_all_seq_5_epochs(data)
    # no_second_boosting(data)
    # no_history_learning_5_epochs(data)
    # no_history_learning_15_epochs(data)
    # no_untemporal_learning(data)
    # no_boosting(data)
    # no_union_learning(data)
    # custom_test(data)
    # common_only_UnionL(data)
    # custom_only_union_l_uncut(data)
    # custom_no_history_l_uncut(data)
    # custom_no_history_l(data)

    unpick = pickle.load(open('2_rooms.pkl', 'rb'))
    dataset = unpick['sparse']
    # unpick2 = pickle.load(open('room2_obs_v1.pkl', 'rb'))
    # v1_sparse2 = unpick2['sparse']
    # states = get_sdr_from_sparse(v1_sparse, unpick['shape'])
    # states2 = get_sdr_from_sparse(v1_sparse2, 180)
    # states_2 = [states, states2]
    # with open('2_rooms.pkl', 'wb') as f:
    #     pickle.dump({
    #         'sparse': states_2,
    #         'shape': 180
    #     }, f)
    # print(states[59].dense & ~states[0].dense)
    states_seqs_test(dataset)


if __name__ == '__main__':
    _run_tests()


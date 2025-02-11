#  Copyright (c) 2024 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Iterable

import numpy as np
from enum import Enum, auto

from hima.common.sdr import sparse_to_dense
from hima.common.utils import softmax, safe_divide
import wandb

EPS = 1e-24


class ExplorationPolicy(Enum):
    SOFTMAX = 1
    EPS_GREEDY = auto()


class ECAgent:
    def __init__(
            self,
            n_obs_states,
            n_actions,
            plan_steps,
            cluster_test_steps,
            minimum_test_steps,
            expand_clusters,
            new_cluster_weight,
            free_state_weight,
            sample_size,
            gamma,
            reward_lr,
            inverse_temp,
            exploration_eps,
            seed
    ):
        self.n_obs_states = n_obs_states
        self.n_actions = n_actions
        self.plan_steps = plan_steps
        self.cluster_test_steps = cluster_test_steps
        self.minimum_test_steps = minimum_test_steps
        self.expand_clusters = expand_clusters
        self.new_cluster_weight = new_cluster_weight
        self.free_state_weight = free_state_weight

        self.first_level_transitions = [dict() for _ in range(n_actions)]
        self.second_level_transitions = [dict() for _ in range(n_actions)]
        self.cluster_to_states = dict()
        self.cluster_to_obs = dict()
        self.state_to_cluster = dict()
        self.obs_to_free_states = {obs: set() for obs in range(self.n_obs_states)}
        self.obs_to_clusters = {obs: set() for obs in range(self.n_obs_states)}

        self.state = (-1, -1)
        self.cluster = {(-1, -1)}
        self.gamma = gamma
        self.reward_lr = reward_lr
        self.rewards = np.zeros(self.n_obs_states, dtype=np.float32)
        self.num_clones = np.zeros(self.n_obs_states, dtype=np.uint32)
        self.action_values = np.zeros(self.n_actions)
        self.first_level_error = 0
        self.second_level_error = 0
        self.first_level_none = 0
        self.second_level_none = 0
        self.sample_size = sample_size
        self.surprise = 0
        self.sf_steps = 0
        self.test_steps = 0
        self.cluster_counter = 0
        self.goal_found = False
        self.learn = True
        self.seed = seed
        self._rng = np.random.default_rng(self.seed)
        self.state_labels = dict()

        self.inverse_temp = inverse_temp
        if exploration_eps < 0:
            self.exploration_policy = ExplorationPolicy.SOFTMAX
            self.exploration_eps = 0
        else:
            self.exploration_policy = ExplorationPolicy.EPS_GREEDY
            self.exploration_eps = exploration_eps

    def reset(self):
        self.state = (-1, -1)
        self.cluster = {(-1, -1)}
        self.goal_found = False
        self.surprise = 0
        self.first_level_error = 0
        self.second_level_error = 0
        self.first_level_none = 0
        self.second_level_none = 0
        self.sf_steps = 0
        self.test_steps = 0
        self.action_values = np.zeros(self.n_actions)

    def observe(self, observation, _reward, learn=True):
        # o_t, a_{t-1}
        obs_state, action = observation
        obs_state = int(obs_state[0])

        predicted_state = self.first_level_transitions[action].get(self.state)
        self.first_level_none = float(predicted_state is None)
        if (predicted_state is None) or (predicted_state[0] != obs_state):
            self.first_level_error = 1
            current_state = self._new_state(obs_state)
        else:
            self.first_level_error = 0
            current_state = predicted_state

        predicted_cluster = set()
        for c in self.cluster:
            pc = self.second_level_transitions[action].get(c)
            if pc is not None:
                predicted_cluster = predicted_cluster.union(pc)
        predicted_obs = {c[0] for c in predicted_cluster}

        self.second_level_none = len(predicted_obs)
        if (len(predicted_cluster) == 0) or not (obs_state in predicted_obs):
            self.second_level_error = 1
            predicted_cluster = {}
        else:
            self.second_level_error = 0

        # state induced cluster
        cluster = self.state_to_cluster.get(current_state)
        if cluster is not None:
            predicted_cluster = {(current_state[0], cluster)}

        current_cluster = predicted_cluster

        if learn:
            if (self.state is not None) and (current_state is not None):
                self.first_level_transitions[action][self.state] = current_state
                if self.state[0] != -1:
                    if cluster is None:
                        self.obs_to_free_states[current_state[0]].add(current_state)

        self.state = current_state
        self.cluster = current_cluster

    def sample_action(self):
        action_values = self.evaluate_actions()
        action_dist = self._get_action_selection_distribution(
            action_values, on_policy=True
        )
        action = self._rng.choice(self.n_actions, p=action_dist)
        return action

    def reinforce(self, reward):
        if self.state is not None:
            obs = self.state[0]
            self.rewards[obs] += self.reward_lr * (
                    reward -
                    self.rewards[obs]
            )

    def evaluate_actions(self):
        self.action_values = np.zeros(self.n_actions)

        planning_steps = 0
        self.goal_found = False
        for action in range(self.n_actions):
            predicted_state = self.first_level_transitions[action].get(self.state)
            sf, steps, gf = self.generate_sf(predicted_state)
            self.goal_found = gf or self.goal_found
            planning_steps += steps
            self.action_values[action] = np.sum(sf * self.rewards)

        self.sf_steps = planning_steps / self.n_actions
        return self.action_values

    def generate_sf(self, initial_state):
        sf = np.zeros(self.n_obs_states, dtype=np.float32)
        goal_found = False

        if initial_state is None:
            return sf, 0, False

        sf[initial_state[0]] = 1

        predicted_states = {initial_state}

        discount = self.gamma

        i = -1
        for i in range(self.plan_steps):
            # uniform strategy
            predicted_states = set().union(
                *[self._predict(predicted_states, a) for a in range(self.n_actions)]
            )
            obs_states = np.array(
                self._convert_to_obs_states(predicted_states),
                dtype=np.uint32
            )
            obs_states, counts = np.unique(obs_states, return_counts=True)
            obs_probs = np.zeros_like(sf)
            obs_probs[obs_states] = counts
            obs_probs /= (obs_probs.sum() + EPS)
            sf += discount * obs_probs

            if len(obs_states) == 0:
                break

            if np.any(
                self.rewards[list(obs_states)] > 0
            ):
                goal_found = True
                break

            discount *= self.gamma

        return sf, i+1, goal_found

    def sleep_phase(self, clustering_iterations):
        self._clustering(clustering_iterations)

    def _clustering(self, iterations):
        if self.num_clones.max() < 2:
            Warning('Interrupting clustering phase. Not enough data.')
            return

        for _ in range(iterations):
            n_free_states = [len(self.obs_to_free_states[obs]) for obs in range(self.n_obs_states)]
            n_free_states = np.array(n_free_states, dtype=np.float32)

            # sample obs state to start replay from
            probs = np.clip(n_free_states - 1, 0, None)
            if probs.sum() == 0:
                probs = np.ones_like(probs)
            probs /= probs.sum()
            obs_state = self._rng.choice(self.n_obs_states, p=probs)

            # sample cluster and states
            # use Chinese-Restaurant-like prior there
            candidates = list(self.obs_to_clusters[obs_state])
            counts = [len(self.cluster_to_states[c]) for c in candidates]

            candidates.append(-1)
            counts.append(self.new_cluster_weight)
            counts = np.array(counts, dtype=np.float32)
            c_probs = counts / (counts.sum() + EPS)

            cluster_id = int(self._rng.choice(candidates, p=c_probs))
            candidates.remove(-1)
            if cluster_id in candidates:
                candidates.remove(cluster_id)

            s_candidates = list(self.obs_to_free_states[obs_state])
            s_free_weights = np.ones(
                len(s_candidates)
            ) * self.free_state_weight

            s_weights = list()
            for c in candidates:
                # sample small clusters more often
                for s in self.cluster_to_states[c]:
                    s_weights.append(1 / len(self.cluster_to_states[c]) ** 2)
                    s_candidates.append(s)
            s_weights = np.array(s_weights)
            s_weights = np.concatenate([s_free_weights, s_weights])
            s_weights /= s_weights.sum()

            sample_size = min(self.sample_size, len(s_candidates))

            if sample_size == 0:
                continue

            candidate_states = self._rng.choice(
                s_candidates,
                sample_size,
                p=s_weights,
                replace=False
            )
            candidate_states = {
                tuple(state) for state in candidate_states
            }
            if cluster_id != -1:
                candidate_states = candidate_states.union(self.cluster_to_states[cluster_id])
            else:
                cluster_id = self.cluster_counter

            old_cluster_assignment = {s: self.state_to_cluster.get(s) for s in candidate_states}

            self._update_cluster(candidate_states, cluster_id, obs_state)

            mask, self.test_steps = self._test_cluster(list(candidate_states))

            succeed_states = {tuple(s) for s, m in zip(candidate_states, mask) if m}
            self._update_cluster(succeed_states, cluster_id, obs_state, old_cluster_assignment)

            if (cluster_id == self.cluster_counter) and (len(succeed_states) > 0):
                self.cluster_counter += 1

            clusters = list(self.cluster_to_states.keys())
            for cluster_id in clusters:
                if len(self.cluster_to_states[cluster_id]) == 0:
                    self._destroy_cluster((self.cluster_to_obs[cluster_id], cluster_id))

            try:
                wandb.log(
                    {
                        'sleep_phase/av_test_steps': self.test_steps,
                        'sleep_phase/num_clusters': self.num_clusters,
                        'sleep_phase/av_cluster_size': self.average_cluster_size,
                        'sleep_phase/num_free_sates': self.num_free_states,
                        'sleep_phase/succeed_states': len(succeed_states),
                        'sleep_phase/clustered_states': len(self.state_to_cluster)
                    }
                )
            except wandb.Error:
                pass

        for cluster_id in self.cluster_to_states:
            # update transition matrix
            for a, d_a in enumerate(self.second_level_transitions):
                predicted_states = [
                    self.first_level_transitions[a][s]
                    for s in self.cluster_to_states[cluster_id]
                    if s in self.first_level_transitions[a]
                ]
                predicted_clusters = [
                    self.state_to_cluster[s] for s in predicted_states
                    if s in self.state_to_cluster
                ]
                predicted_clusters, counts = np.unique(predicted_clusters, return_counts=True)
                assert len(set([self.cluster_to_obs[c] for c in predicted_clusters])) <= 1
                if len(counts) > 0:
                    cluster = (self.cluster_to_obs[cluster_id], cluster_id)
                    pred_clusters = {(self.cluster_to_obs[pc], pc) for pc in predicted_clusters}
                    d_a[cluster] = pred_clusters

    def _rehearse(self, iterations):
        pass

    def _update_cluster(
            self,
            new_states: set,
            cluster_id: int,
            obs_state: int,
            old_cluster_assignment: dict=None
    ):
        """
            Update cluster state diff
            Assign new_states to another cluster and also update clusters that are affected.

            old_cluster_assignment: dict
        """
        for s in new_states:
            if s in self.state_to_cluster:
                # remove from old cluster
                old_c = self.state_to_cluster[s]
                if old_c != cluster_id:
                    self.cluster_to_states[old_c].remove(s)
            else:
                self.obs_to_free_states[obs_state].remove(s)
            self.state_to_cluster[s] = cluster_id

        if cluster_id in self.cluster_to_states:
            removed_states = self.cluster_to_states[cluster_id].difference(new_states)
            for s in removed_states:
                old_cluster = None
                if old_cluster_assignment is not None:
                    old_cluster = old_cluster_assignment.get(s)
                if (old_cluster is not None) and (old_cluster != cluster_id):
                    self.state_to_cluster[s] = old_cluster
                    self.cluster_to_states[old_cluster].add(s)
                else:
                    if s in self.state_to_cluster:
                        self.state_to_cluster.pop(s)
                    self.obs_to_free_states[obs_state].add(s)

        self.cluster_to_states[cluster_id] = new_states
        self.cluster_to_obs[cluster_id] = obs_state
        self.obs_to_clusters[obs_state].add(cluster_id)

    def _destroy_cluster(self, cluster: tuple):
        # TODO free states
        obs_state, cluster_id = cluster
        if cluster_id in self.cluster_to_states:
            self.cluster_to_states.pop(cluster_id)

        if obs_state in self.obs_to_clusters:
            self.obs_to_clusters[obs_state].remove(cluster_id)

        for d_a in self.second_level_transitions:
            if cluster in d_a:
                d_a.pop(cluster)

    def _test_cluster(self, cluster: list) -> (np.ndarray, int):
        """
            Returns boolean array of size len(cluster)
            True/False means that the cluster's element is
                consistent/inconsistent with the majority of elements
        """
        ps_per_i = {pos: {s} for pos, s in enumerate(cluster)}
        test = np.ones(len(ps_per_i)).astype(np.bool8)
        t = -1
        for t in range(self.cluster_test_steps):
            score_a = np.zeros(self.n_actions)
            # predict states for each action and initial state
            ps_per_a = [dict() for _ in range(self.n_actions)]
            trace_interrupted = np.ones_like(test)
            for a, d_a in enumerate(self.first_level_transitions):
                obs = np.full(len(cluster), fill_value=np.nan)
                for pos, ps_i in ps_per_i.items():
                    ps_a = self._predict(ps_i, a, expand_clusters=(t!=0) and self.expand_clusters)
                    obs_a = {s[0] for s in ps_a}
                    # contradiction
                    if len(obs_a) > 1:
                        test[pos] = False
                    elif len(obs_a) == 1:
                        score_a[a] += int(len(ps_a) != 0)
                        ps_per_a[a][pos] = ps_a
                        obs[pos] = obs_a.pop()
                        trace_interrupted[pos] = False
                # detect contradiction
                empty = np.isnan(obs)
                states, counts = np.unique(obs[~empty], return_counts=True)
                if len(counts) > 0:
                    test = test & ((obs == states[np.argmax(counts)]) | empty)

            # discard traces that were interrupted too early
            if t < self.minimum_test_steps:
                test = test & (~trace_interrupted)

            # choose next action
            action = np.argmax(score_a)
            ps_per_i = {pos: s for pos, s in ps_per_a[action].items() if test[pos]}

            if (score_a[action] <= 1) or (len(ps_per_i) <= 1):
                # no predictions or only one trace is left
                break

        return test, t+1

    def _predict(self, state: set, action: int, expand_clusters: bool = False) -> set:
        state_expanded = state
        if expand_clusters:
            clusters = [self.state_to_cluster.get(s) for s in state]
            state_expanded = state_expanded.union(
                *[self.cluster_to_states[c] for c in clusters if c is not None]
            )

        d_a = self.first_level_transitions[action]
        predicted_state = set()
        for s in state_expanded:
            if s in d_a:
                predicted_state.add(d_a[s])
        return predicted_state

    def _convert_to_obs_states(self, states):
        return [s[0] for s in states if s is not None]

    def _new_state(self, obs_state):
        h = int(self.num_clones[obs_state])
        self.num_clones[obs_state] += 1
        return obs_state, h

    def _get_action_selection_distribution(
            self, action_values, on_policy: bool = True
    ) -> np.ndarray:
        # off policy means greedy, on policy — with current exploration strategy
        if on_policy and self.exploration_policy == ExplorationPolicy.SOFTMAX:
            # normalize values before applying softmax to make the choice
            # of the softmax temperature scale invariant
            action_values = safe_divide(action_values, np.abs(action_values.sum()))
            action_dist = softmax(action_values, beta=self.inverse_temp)
        else:
            # greedy off policy or eps-greedy
            best_action = np.argmax(action_values)
            # make greedy policy
            # noinspection PyTypeChecker
            action_dist = sparse_to_dense([best_action], like=action_values)

            if on_policy and self.exploration_policy == ExplorationPolicy.EPS_GREEDY:
                # add uniform exploration
                action_dist[best_action] = 1 - self.exploration_eps
                action_dist[:] += self.exploration_eps / self.n_actions

        return action_dist

    @property
    def total_num_clones(self):
        return self.num_clones.sum()

    @property
    def num_clusters(self):
        return len(self.cluster_to_states)

    @property
    def num_free_states(self):
        return sum([len(v) for v in self.obs_to_free_states.values()])

    @property
    def average_cluster_size(self):
        return np.array([len(self.cluster_to_states[c]) for c in self.cluster_to_states]).mean()

    @property
    def num_transitions_second_level(self):
        return sum([len(d_a) for d_a in self.second_level_transitions])

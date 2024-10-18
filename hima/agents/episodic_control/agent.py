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

        self.first_level_transitions = [dict() for _ in range(n_actions)]
        self.second_level_transitions = [dict() for _ in range(n_actions)]
        self.cluster_to_states = dict()
        self.state_to_cluster = dict()
        self.obs_to_free_states = {obs: set() for obs in range(self.n_obs_states)}
        self.obs_to_clusters = {obs: set() for obs in range(self.n_obs_states)}

        self.state = (0, 0)
        self.gamma = gamma
        self.reward_lr = reward_lr
        self.rewards = np.zeros(self.n_obs_states, dtype=np.float32)
        self.num_clones = np.zeros(self.n_obs_states, dtype=np.uint32)
        self.action_values = np.zeros(self.n_actions)
        self.surprise = 0
        self.sf_steps = 0
        self.test_steps = 0
        self.cluster_counter = 0
        self.goal_found = False
        self.seed = seed
        self._rng = np.random.default_rng(self.seed)

        self.inverse_temp = inverse_temp
        if exploration_eps < 0:
            self.exploration_policy = ExplorationPolicy.SOFTMAX
            self.exploration_eps = 0
        else:
            self.exploration_policy = ExplorationPolicy.EPS_GREEDY
            self.exploration_eps = exploration_eps

    def reset(self):
        self.state = (0, 0)
        self.goal_found = False
        self.surprise = 0
        self.sf_steps = 0
        self.test_steps = 0
        self.action_values = np.zeros(self.n_actions)

    def observe(self, observation, _reward, learn=True):
        # o_t, a_{t-1}
        obs_state, action = observation
        obs_state = int(obs_state[0])

        predicted_state = self.first_level_transitions[action].get(self.state)
        if (predicted_state is None) or (predicted_state[0] != obs_state):
            current_state = self._new_state(obs_state)
        else:
            current_state = predicted_state

        if learn:
            self.first_level_transitions[action][self.state] = current_state
            self.obs_to_free_states[self.state[0]].add(self.state)

        self.state = current_state

    def sample_action(self):
        action_values = self.evaluate_actions()
        action_dist = self._get_action_selection_distribution(
            action_values, on_policy=True
        )
        action = self._rng.choice(self.n_actions, p=action_dist)
        return action

    def reinforce(self, reward):
        self.rewards[self.state[0]] += self.reward_lr * (
                reward -
                self.rewards[self.state[0]]
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

    def sleep_phase(self, sleep_iterations):
        for _ in range(sleep_iterations):
            n_free_states = [len(self.obs_to_free_states[obs]) for obs in range(self.n_obs_states)]
            n_free_states = np.array(n_free_states, dtype=np.float32)

            if n_free_states.max() < 2:
                Warning('Interrupting sleep phase. Not enough data.')
                return

            # sample obs state to start replay from
            probs = np.clip(n_free_states - 1, 0, None)
            probs /= probs.sum()
            obs_state = self._rng.choice(self.n_obs_states, p=probs)

            # TODO add cluster merging
            # sample cluster and states
            candidates = list(self.obs_to_clusters[obs_state])
            candidates.append(-1)
            cluster_id = int(self._rng.choice(candidates))
            if cluster_id == -1:
                # create new cluster
                cluster_states = {
                    tuple(state) for state in
                    self._rng.choice(
                        list(self.obs_to_free_states[obs_state]),
                        2,
                        replace=False
                    )
                }
            else:
                cluster_states = self.cluster_to_states[cluster_id]
                new_state = tuple(self._rng.choice(list(self.obs_to_free_states[obs_state])))
                cluster_states.add(new_state)

            mask, self.test_steps = self._test_cluster(cluster_states)
            cluster_states = np.array(list(cluster_states))
            freed_states = cluster_states[~mask]
            cluster_states = cluster_states[mask]
            freed_states = {tuple(s) for s in freed_states}
            cluster_states = {tuple(s) for s in cluster_states}

            # update cluster
            if (len(cluster_states) < 2) and cluster_id == -1:
                # cluster failed to form, nothing to change
                return

            if len(cluster_states) < 2:
                # cluster destroyed
                freed_states = self.cluster_to_states.pop(cluster_id)
                self.obs_to_clusters[obs_state].remove(cluster_id)
            else:
                if cluster_id == -1:
                    cluster_id = self.cluster_counter
                    self.cluster_counter += 1
                    self.obs_to_clusters[obs_state].add(cluster_id)

                self.cluster_to_states[cluster_id] = cluster_states
                for c in cluster_states:
                    self.state_to_cluster[c] = cluster_id
                self.obs_to_free_states[obs_state] = self.obs_to_free_states[obs_state].difference(
                    self.cluster_to_states[cluster_id]
                )

            # update free states
            self.obs_to_free_states[obs_state].update(freed_states)
            for s in freed_states:
                if s in self.state_to_cluster:
                    self.state_to_cluster.pop(s)

    def _test_cluster(self, cluster: Iterable) -> (np.ndarray, int):
        """
            Returns boolean array of size len(cluster)
            True/False means that the cluster's element is
                consistent/inconsistent with the majority of elements
        """
        ps_per_i = [{s} for s in cluster]
        t = -1
        for t in range(self.cluster_test_steps):
            score_a = np.zeros(self.n_actions)
            # predict states for each action and initial state
            ps_per_a = [[] for _ in range(self.n_actions)]
            obs_per_a = [[] for _ in range(self.n_actions)]
            for a, d_a in enumerate(self.first_level_transitions):
                for ps_i in ps_per_i:
                    ps_a = self._predict(ps_i, a)
                    if len(ps_a) > 0:
                        score_a[a] += 1
                        obs_a = self._convert_to_obs_states(ps_a)
                        obs_a = set(obs_a)
                        if len(obs_a) > 1:
                            obs_a = -1
                        else:
                            obs_a = obs_a.pop()
                    else:
                        obs_a = np.nan

                    ps_per_a[a].append(ps_a)
                    obs_per_a[a].append(obs_a)

                # detect contradiction
                obs = np.array(obs_per_a[a])
                empty = np.isnan(obs)
                states, counts = np.unique(obs[~empty], return_counts=True)
                if len(states) > 1:
                    test = (obs == states[np.argmax(counts)]) | empty
                    return test, t+1
            # choose next action
            action = np.argmax(score_a)
            ps_per_i = ps_per_a[action]
            obs = obs_per_a[action]

            if score_a[action] <= 1:
                # no predictions
                break

            obs = np.array(obs)
            obs = obs[~np.isnan(obs)]
            if len(obs) > 0:
                if np.any(
                        self.rewards[obs.astype(np.int32)] > 0
                ):
                    # found rewarding state
                    break

        return np.ones(len(ps_per_i)).astype(np.bool8), t+1

    def _predict(self, state: set, action: int) -> set:
        clusters = [self.state_to_cluster.get(s) for s in state]
        state_expanded = set().union(
            *[self.cluster_to_states[c] for c in clusters if c is not None]
        )
        state_expanded.update(state)
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
    def average_cluster_size(self):
        return np.array([len(self.cluster_to_states[c]) for c in self.cluster_to_states]).mean()

import numpy as np
from enum import Enum, auto
from hima.modules.eprop_rnn.eprop_rnn import RNNWithEProp
from hima.modules.eprop_rnn.constants import EPS
from hima.common.utils import safe_divide, softmax
from hima.common.sdr import sparse_to_dense
from hima.experiments.tpcn.runner.transition_matrix import build_transition_matrix, compute_metrics

class ExplorationPolicy(Enum):
    SOFTMAX = 1
    EPS_GREEDY = auto()

class RNNWithEPropAgent:
    def __init__(self,
                 model_config: dict,
                 n_obs_states: int,
                 n_actions: int,
                 batch_size: int = 64,
                 learning_rate: float = 0.005,
                 log_transitions: bool = False,
                 gamma: float = 0.9,
                 learn: bool = True,
                 reward_lr: float = 0.01,
                 learn_rewards_from_state: bool = True,
                 plan_steps: int = 1,
                 inverse_temp: float = 1.0,
                 exploration_eps: float = -1,
                 sr_estimate_planning: str = 'uniform',
                 sr_early_stop_uniform: float | None = None,
                 sr_early_stop_goal: float | None = None,
                 seed: int = 42):
        
        self.lr = learning_rate
        self.cum_reward = 0
        self.n_actions = n_actions
        self.n_obs_states = n_obs_states
        self.exploration_eps = exploration_eps
        self.observations = []
        self.actions = []
        self.concat_vecs = [[]]
        self.episode = 0
        self.learn = learn
        self.batch_size = batch_size
        self.inverse_temp = inverse_temp
        self.plan_steps = plan_steps
        self.seed = seed
        self.sr_estimate_planning = sr_estimate_planning
        self.sr_early_stop_uniform = sr_early_stop_uniform
        self.sr_early_stop_goal = sr_early_stop_goal
        self.gamma = gamma
        self.reward_lr = reward_lr
        self.model_config = model_config
        self.sf_steps = 0
        self.is_first = True
        
        self.model_config['output_size'] = self.n_obs_states

        self.model = RNNWithEProp(**self.model_config)

        self.learn_rewards_from_state = learn_rewards_from_state
        if self.learn_rewards_from_state:
            rewards = np.zeros((self.model.hidden_size))
        else:
            rewards = np.zeros((self.n_obs_states))
        
        self.rewards = rewards.flatten()

        self.true_transition_matrix = None
        self.transition_matrix = None
        self.spectral_diff = 0
        self.entropy_diff = 0
        self.jensenshannon_diff = 0
        self.log_transitions = log_transitions

        self.onehot_obs = []
        self.prev_hidden = np.zeros((1, self.model.hidden_size))
        self.action = None
        self.hiddens = []
        self.batch_ind = []
        self.preds = []

        if self.exploration_eps < 0:
            self.exploration_policy = ExplorationPolicy.SOFTMAX
            self.exploration_eps = 0
        else:
            self.exploration_policy = ExplorationPolicy.EPS_GREEDY
            self.exploration_eps = self.exploration_eps
        self._rng = np.random.default_rng(seed=seed)

    def observe(self, obs, action, reward):
        encoded_obs, onehot_obs = obs
        self.onehot_obs.append(onehot_obs.reshape(1, -1))
        if not self.is_first:
            self.actions.append(action.reshape(1, -1))
            self.concat_vecs.append(np.concatenate((self.observations[-1], self.actions[-1]), axis=-1))
            self.prev_hidden, pred = self.model.forward(self.concat_vecs[-1])
            self.preds.append(pred.reshape(1, -1))
        else:
            self.is_first = False
            self.preds.append(np.zeros(self.n_obs_states).reshape(1, -1))
        self.observations.append(encoded_obs.reshape(1, -1))

    
    def predict(self, x):
        return self.model.decode(self.prev_hidden)
    
    def get_state_representation(self):
        return self.prev_hidden.squueze()

    def generate_sf(self, init_hidden, initial_prediction, n_steps=5, gamma=0.95, learn_sr=False):
        
        hiddens = [init_hidden]
        sr = init_hidden
        predictions = [initial_prediction]
        sf = initial_prediction
        discounts = [1.0]
        actions = np.ones(self.n_actions) / self.n_actions
        self.sf_steps = 1
        for _ in range(n_steps):
            if self.learn_rewards_from_state:
                early_stop = self._early_stop_planning(
                    sr.reshape(-1, 1)
                )
            else:
                early_stop = self._early_stop_planning(
                    sf.reshape(-1, 1)
                )
            
            predictions.append(self.model.predict_one(np.concat((predictions[-1], actions))))
            hiddens.append(self.model.hidden_states[-1])

            if learn_sr:
                sr += discounts[-1] * hiddens[-1]

            sf += discounts[-1] * self.model.decode(hiddens[-1])

            discounts.append(discounts[-1] * gamma)

            self.sf_steps += 1
            
            if early_stop:
                break
        
        if learn_sr:
            return sf, sr
        else:
            return sf

    def sample_action(self):
        """Evaluate and sample actions."""
        self.action_values = self.evaluate_actions(n_steps=self.plan_steps, gamma=self.gamma)
        self.action_dist = self._get_action_selection_distribution(
            self.action_values, on_policy=True
        )

        self.action = self._rng.choice(self.n_actions, p=self.action_dist)
        self.encoded_action = np.zeros(self.n_actions, dtype=np.float32)
        self.encoded_action[self.action] = 1.0
        self.encoded_action = self.encoded_action
        return self.action 

    def reset(self):
        if self.episode > 0:
            self.actions.append(np.zeros(self.n_actions).reshape(1, -1))
            self.concat_vecs = np.vstack(self.concat_vecs)
            self.hiddens.extend(self.model.hidden_states)
            self.model.train_step(self.concat_vecs)
        
            if self.episode % self.batch_size == 0:
                if self.log_transitions:
                    self.batch_ind.append(len(self.observations))
                    self.observations = np.concatenate(self.observations)
                    self.actions = np.concatenate(self.actions)
                    self.hiddens = np.concatenate(self.hiddens)
                    self.onehot_obs = np.concatenate(self.onehot_obs)
                    self.transition_matrix = build_transition_matrix(
                        states=self.hiddens,
                        labels = self.onehot_obs.argmax(axis=-1),
                        n_clusters=self.true_transition_matrix.shape[1],
                        actions=self.actions.argmax(axis=-1),
                        batch_idx=self.batch_ind,
                        n_actions=self.n_actions
                    )
                    self.spectral_diff, self.jensenshannon_diff, self.entropy_diff = compute_metrics(
                        transition_matrix=self.transition_matrix,
                        true_transition_matrix=self.true_transition_matrix
                    )
                self.preds = np.argmax(np.vstack(self.preds), -1)
                trues = np.argmax(self.onehot_obs, -1)
                acc = (trues == self.preds).mean()
                print("Accuracy:", acc)
                
                self.preds = []
                self.onehot_obs = []
                self.hiddens = []
                self.batch_ind = []
                self.observations = []
                self.actions = []
            else:
                self.batch_ind.append(len(self.observations))
        
        self.model.reset_states()
        self.episode += 1
        self.concat_vecs = []
        self.cum_reward = 0
        self.prev_hidden = np.zeros((1, self.model.hidden_size))
        self.action = None
        self.is_first = True
    
    def reinforce(self, reward):
        if self.learn_rewards_from_state:
            deltas = self.prev_hidden.squeeze() * (reward - self.rewards)
        else:
            deltas = self.model.decode(self.prev_hidden).squeeze() * (reward - self.rewards)
        self.rewards += self.reward_lr * deltas

    def evaluate_actions(self, n_steps=5, gamma=0.95):
        
        n_actions = self.n_actions
        action_values = np.zeros(n_actions)
        dense_action = np.zeros_like(action_values)
        snapshot = self.model.hidden_states

        for action in range(n_actions):
            dense_action[action - 1] = 0
            dense_action[action] = 1

            prediction = self.model.decode(self.prev_hidden)
            pred_hidden = self.prev_hidden
            if self.learn_rewards_from_state:
                sf, sr = self.generate_sf(
                    initial_prediction=prediction,
                    init_hidden=pred_hidden, 
                    n_steps=n_steps, 
                    gamma=gamma,
                    learn_sr=True)

                action_values[action] = np.sum(
                        sr * self.rewards
                    ) / self.model.hidden_size
            else:
                sf = self.generate_sf(
                initial_prediction=prediction,
                init_hidden=pred_hidden, 
                n_steps=n_steps, 
                gamma=gamma)

                action_values[action] = np.sum(
                        sf * self.rewards
                    ) / self.n_obs_states

            self.model.hidden_states = snapshot
        return action_values

    def _early_stop_planning(self, states: np.ndarray) -> bool:
        n_vars, n_states = states.shape

        if self.sr_early_stop_uniform is not None:
            uni_dkl = (
                    np.log(n_states) +
                    np.sum(
                        states * np.log(
                            np.clip(
                                states, EPS, None
                            )
                        ),
                        axis=-1
                    )
            )

            uniform = uni_dkl.mean() < self.sr_early_stop_uniform
        else:
            uniform = False

        if self.sr_early_stop_goal is not None:
            goal = np.any(
                np.sum(
                    (states.flatten() * (self.rewards > 0)).reshape(
                        n_vars, -1
                    ),
                    axis=-1
                ) > self.sr_early_stop_goal
            )
        else:
            goal = False

        return uniform or goal

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
    

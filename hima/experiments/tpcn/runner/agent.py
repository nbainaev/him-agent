import numpy as np
from hima.agents.tpcn.agent import tPCNAgent
from hima.experiments.successor_representations.runners.base import BaseAgent
from hima.agents.eprop_rnn.agent import RNNWithEPropAgent

class TPCNWrapper(BaseAgent):
    def __init__(self, conf):
        
        self.conf = conf
        self.encoder_type = conf['encoder_type']

        self.initial_action = 0
        self.logger = None
        self.n_actions = conf['agent']['n_actions']
        self.n_obs_states = conf['agent']['n_obs_states']
        self.conf['agent']['model_config']['n_obs_states'] = self.conf['agent']['n_obs_states']
        self.conf['agent']['model_config']['hidden_size'] = self.conf['agent']['hidden_size']
        self.conf['agent']['model_config']['n_actions'] = self.conf['agent']['n_actions']

        self.conf['agent']['log_transitions'] = self.conf.pop('log_transitions')
        self.agent = tPCNAgent(**self.conf['agent'])
        if self.conf['weights'] is not None:
            import torch
            model_weights = torch.load(self.conf['weights'])
            self.agent.model.load_state_dict(model_weights)
        
        self.agent.true_transition_matrix = conf.pop('transition_matrix')

        self.encoder = self._make_encoder()

        self.init_acc = []

    def observe(self, obs, action, reward=0):
        encoded_action = np.zeros(self.n_actions)
        encoded_action[action] = 1
        return self.agent.observe(
            self.encoder.encode(obs),
            encoded_action,
            reward
        )
    
    def sample_action(self):
        return self.agent.sample_action()

    def reinforce(self, reward):
        return self.agent.reinforce(reward)
    
    def reset(self):
        return self.agent.reset()

    
    @property
    def state_value(self):
        action_values = self.agent.action_values
        if action_values is None:
            action_values = self.agent.evaluate_actions()
        state_value = np.sum(action_values)
        return state_value

    @property
    def action_values(self):
        action_values = self.agent.action_values
        if action_values is None:
            action_values = self.agent.evaluate_actions()
        return action_values
    
    @property
    def state_repr(self):
        return self.agent.prev_hidden.detach().numpy().squueze()

    def _make_encoder(self):
            if self.encoder_type == 'sp_grouped':
                from hima.modules.belief.cortial_column.encoders.sp import (
                    SpatialPooler
                )

                encoder_conf = self.conf['encoder']
                encoder_conf['seed'] = self.seed
                encoder_conf['feedforward_sds'] = [self.n_obs_states, 0.1]

                encoder = SpatialPooler(encoder_conf)
            elif self.encoder_type == 'pcn':
                from encoders import HierarchicalPCN
                encoder_conf = self.conf['encoder']
                encoder_conf['n_obs_states'] = self.n_obs_states
                encoder_conf['hidden_size'] = self.conf['agent']['hidden_size']
                encoder = HierarchicalPCN(**encoder_conf)
                if self.conf['encoder_weights'] is not None:
                    import torch
                    encoder_weigths = torch.load(self.conf['encoder_weights'])
                    encoder.load_state_dict(encoder_weigths['pcn'])
                    encoder.onehot_encoder.categories = encoder_weigths['onehot']['categories']
                    encoder.onehot_encoder.max_categories = encoder_weigths['onehot']['max_categories']
            elif self.encoder_type == 'onehot':
                from encoders import GridWorldOnehotEncoder
                encoder = GridWorldOnehotEncoder(
                    max_categories=self.n_obs_states, 
                    out_dim=self.conf['agent']['model_config']['hidden_size']
                )
                encoder.categories = dict([(i, i) for i in range(self.n_obs_states)])
            elif self.encoder_type == 'vae':
                from hima.modules.belief.cortial_column.encoders.vae import CatVAE
                encoder = CatVAE(**self.conf['encoder'])
            elif self.encoder_type == 'kmeans':
                from hima.modules.belief.cortial_column.encoders.kmeans import KMeansEncoder
                encoder = KMeansEncoder(**self.conf['encoder'])
            elif self.encoder_type is None:
                encoder = None
            else:
                raise ValueError(f'Encoder type {self.encoder_type} is not supported')
            
            return encoder
    
class RNNWithEPropWrapper(BaseAgent):
    def __init__(self, conf):
        
        self.conf = conf
        self.encoder_type = conf['encoder_type']
        
        self.initial_action = 0
        self.logger = None
        self.n_actions = conf['agent']['n_actions']
        self.n_obs_states = conf['agent']['n_obs_states']
        self.conf['agent']['model_config']['output_size'] = self.n_obs_states

        self.conf['agent']['log_transitions'] = self.conf.pop('log_transitions')
        self.agent = RNNWithEPropAgent(**self.conf['agent'])
        if self.conf['weights'] is not None:
            model_weights = np.load(self.conf['weights'])
            self.agent.model.W_hh = model_weights['W_hh']
            self.agent.model.W_hy = model_weights['W_hy']
            self.agent.model.W_xh = model_weights['W_xh']
            self.agent.model.b_h = model_weights['b_h']
            self.agent.model.b_y = model_weights['b_y']

        self.agent.true_transition_matrix = conf.pop('transition_matrix')
        
        self.encoder = self._make_encoder()

    def observe(self, obs, action, reward=0):
        encoded_action = np.zeros(self.n_actions)
        encoded_action[action] = 1
        
        return self.agent.observe(
            self.encoder.encode(obs), 
            encoded_action,
            reward
        )
    
    def sample_action(self):
        return self.agent.sample_action()

    def reinforce(self, reward):
        return self.agent.reinforce(reward)
    
    def reset(self):
        return self.agent.reset()

    
    @property
    def state_value(self):
        action_values = self.agent.action_values
        if action_values is None:
            action_values = self.agent.evaluate_actions()
        state_value = np.sum(action_values)
        return state_value

    @property
    def action_values(self):
        action_values = self.agent.action_values
        if action_values is None:
            action_values = self.agent.evaluate_actions()
        return action_values
    
    @property
    def state_repr(self):
        return self.agent.prev_hidden.squueze()

    def _make_encoder(self):
            if self.encoder_type == 'sp_grouped':
                from hima.modules.belief.cortial_column.encoders.sp import (
                    SpatialPooler
                )

                encoder_conf = self.conf['encoder']
                encoder_conf['seed'] = self.seed
                encoder_conf['feedforward_sds'] = [self.n_obs_states, 0.1]

                encoder = SpatialPooler(encoder_conf)
            elif self.encoder_type == 'pcn':
                from encoders import HierarchicalPCN
                encoder_conf = self.conf['encoder']
                encoder_conf['n_obs_states'] = self.n_obs_states
                encoder_conf['hidden_size'] = self.conf['agent']['hidden_size']
                encoder = HierarchicalPCN(**encoder_conf)
                if self.conf['encoder_weights'] is not None:
                    import torch
                    encoder_weigths = torch.load(self.conf['encoder_weights'])
                    encoder.load_state_dict(encoder_weigths['pcn'])
                    encoder.onehot_encoder.categories = encoder_weigths['onehot']['categories']
                    encoder.onehot_encoder.max_categories = encoder_weigths['onehot']['max_categories']
            elif self.encoder_type == 'onehot':
                from encoders import GridWorldOnehotEncoder
                encoder = GridWorldOnehotEncoder(
                    max_categories=self.n_obs_states, 
                    out_dim=self.conf['agent']['model_config']['input_size'] - self.n_actions
                )
                encoder.categories = dict([(i, i) for i in range(self.n_obs_states)])
            elif self.encoder_type == 'vae':
                from hima.modules.belief.cortial_column.encoders.vae import CatVAE
                encoder = CatVAE(**self.conf['encoder'])
            elif self.encoder_type == 'kmeans':
                from hima.modules.belief.cortial_column.encoders.kmeans import KMeansEncoder
                encoder = KMeansEncoder(**self.conf['encoder'])
            elif self.encoder_type is None:
                encoder = None
            else:
                raise ValueError(f'Encoder type {self.encoder_type} is not supported')
            
            return encoder
    

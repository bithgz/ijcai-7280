from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import copy
import os
from network.fme_c_net import SquashedGaussianMLPActor, MLPQFunction
from collections import OrderedDict

class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes=(256,256), activation=nn.ReLU):
        super().__init__()
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.qh1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.qh2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.cpu().numpy()



class FME_C(nn.Module):

    def __init__(self, args, is_cuda):
        super().__init__()
        self.args = args
        self.n_agents_per_party = args.n_agents_per_party
        self.obs_size = args.obs_shape
        self.action_size = args.n_actions
        self.act_limit = args.act_limit
        
        self.gamma = self.args.gamma
        self.hidden_size = args.rnn_hidden_dim
        self.learning_rate = args.lr
        self.clip_grad_param = args.grad_norm_clip

        self.alpha = args.alpha
        self.is_cuda = is_cuda
        
        self.model_dir = self.args.save_path

        self.ac = MLPActorCritic(self.obs_size, self.action_size, self.act_limit, hidden_sizes=(self.hidden_size, self.hidden_size))
        self.ac_targ = deepcopy(self.ac) 
        

        if self.is_cuda:
            self.ac.cuda()
            self.ac_targ.cuda()

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
            
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters(), 
                                        self.ac.qh1.parameters(), self.ac.qh2.parameters())
       
        
        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.learning_rate)
        self.q_optimizer = Adam(self.q_params, lr=self.learning_rate)
        # self.qh_optimizer = Adam(self.qh_params, lr=self.learning_rate)



    def get_para(self):
        device = torch.device('cpu')

        state_dict = self.ac.pi.state_dict()

        new_state_dict = OrderedDict()

        for k, v in state_dict.items():

            name = k # remove `module.`

            new_state_dict[name] = v.to(device)
        # model.load_state_dict(new_state_dict)
        return new_state_dict


    def load_para(self, para):
        self.ac.pi.load_state_dict(para)


    def _trans_shape(self, obs):
        transition_num = obs.shape[0]
        inputs = []
        inputs.append(obs)
        inputs = torch.cat([x.reshape(transition_num * 2 * self.args.n_agents_per_party, -1) for x in inputs], dim=1)
        return inputs




    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, batch):
        states = batch['o'] # 1000,2,aa
        inputs = self._trans_shape(states) # 2000,aa
    
        pi, logp_pi = self.ac.pi(inputs)  # 2000,
     
        q1_pi = self.ac.q1(inputs, pi) # 2000,
        q2_pi = self.ac.q2(inputs, pi) # 2000,
        q_pi = torch.min(q1_pi, q2_pi) # 2000,
       
        qh1_pi = self.ac.qh1(inputs, pi) # 2000,
        qh2_pi = self.ac.qh2(inputs, pi) # 2000,
        qh_pi = torch.min(qh1_pi, qh2_pi) # 2000,
    

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - (q_pi + self.alpha * qh_pi)).mean()
    
        # Useful info for logging
        # pi_info = dict(LogPi=logp_pi.detach().numpy())
        return loss_pi
    


    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, batch):
        transition_num = batch['o'].shape[0]
      
        states, next_states, actions, rewards, dones = batch['o'], batch['o_next'], batch['u'], \
                                                             batch['r'], batch['terminated']

        with torch.no_grad():
            
            next_inputs = self._trans_shape(next_states)
            next_actions, logp_next_actions = self.ac.pi(next_inputs)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(next_inputs, next_actions) # 2000,
            q2_pi_targ = self.ac_targ.q2(next_inputs, next_actions) # 2000,
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)  # 2000,
            q_pi_targ = q_pi_targ.view(transition_num, 2 * self.n_agents_per_party) # 1000, 2
            q_pi_targ_p = torch.split(q_pi_targ, self.n_agents_per_party, 1)[0] # 1000, 1
            q_pi_targ_n = torch.neg(torch.split(q_pi_targ, self.n_agents_per_party, 1)[1]) # 1000, 1
            q_pi_targ_next = q_pi_targ_p.sum(1).unsqueeze(-1) + q_pi_targ_n.sum(1).unsqueeze(-1) # 1000, 1
            Q_targets = rewards + (self.gamma * (1 - dones) * q_pi_targ_next) # 1000, 1

            # Target Qh-values
            qh1_pi_targ = self.ac_targ.qh1(next_inputs, next_actions) # 2000,
            qh2_pi_targ = self.ac_targ.qh2(next_inputs, next_actions) # 2000,
            qh_pi_targ = torch.min(qh1_pi_targ, qh2_pi_targ) # 2000,
            Q_h_targets = self.gamma * (-logp_next_actions) + self.gamma \
                * (1 - dones.squeeze(-1).repeat(2 * self.n_agents_per_party)) * qh_pi_targ    


        # Compute critic loss
        inputs = self._trans_shape(states)
        actions = self._trans_shape(actions)

        q1 = self.ac.q1(inputs, actions)# 2000,
        q2 = self.ac.q2(inputs, actions) # 2000,
        min_q = torch.min(q1, q2)
        q = min_q.view(transition_num, 2 * self.n_agents_per_party)#----1000,2
        q_p = torch.split(q, self.n_agents_per_party, 1)[0]
        q_n = torch.neg(torch.split(q, self.n_agents_per_party, 1)[1])
        q_tot = q_p.sum(1).unsqueeze(-1) + q_n.sum(1).unsqueeze(-1)
        critic_loss = 0.5 * F.mse_loss(q_tot, Q_targets)

        q_h1 = self.ac.qh1(inputs, actions)# 2000,
        q_h2 = self.ac.qh2(inputs, actions)# 2000,
        min_q_h = torch.min(q_h1, q_h2)
        critic_h_loss = 0.5 * F.mse_loss(min_q_h, Q_h_targets)

        return critic_loss + critic_h_loss

       

    def train_critic_and_actor(self, batch, train_step, target_update):
    
        # transition_num = batch['o'].shape[0]
        if self.is_cuda:
            for key in batch.keys():  # 把batch里的数据转化成tensor
                batch[key] = torch.tensor(batch[key], dtype=torch.float32).cuda()
        else:
            for key in batch.keys():  # 把batch里的数据转化成tensor
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
       

        # ---------------------------- update actor ---------------------------- #
        loss_pi = self.compute_loss_pi(batch)
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        clip_grad_norm_(self.ac.pi.parameters(), self.clip_grad_param)
        self.pi_optimizer.step()
        # ---------------------------- update Q ---------------------------- #
        loss_q = self.compute_loss_q(batch)
        self.q_optimizer.zero_grad()
        loss_q.backward()
        clip_grad_norm_(self.q_params, self.clip_grad_param)
        self.q_optimizer.step()
        # ----------------------- update target networks ----------------------- #
        if train_step == 0 and target_update:
            self.hard_update(self.ac.pi, self.ac_targ.pi)
            self.hard_update(self.ac.q1, self.ac_targ.q1)
            self.hard_update(self.ac.q2, self.ac_targ.q2)
            self.hard_update(self.ac.qh1, self.ac_targ.qh1)
            self.hard_update(self.ac.qh2, self.ac_targ.qh2)



    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)


    def save_model(self, eps):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.ac.pi.state_dict(),  self.model_dir + '/' + str(eps) + '_actor_net_params.pkl')

    def load_model(self, eps):
        if os.path.exists(self.model_dir + '/' + str(eps) + '_actor_net_params.pkl'):
            path_actor = self.model_dir + '/' + str(eps) + '_actor_net_params.pkl'
            map_location = 'cuda:'+ self.args.CUDA_VISIBLE_DEVICES
            self.ac.pi.load_state_dict(torch.load(path_actor, map_location=map_location))
            print('Successfully load the model: {}'.format(path_actor))
        else:
            raise Exception("No model!")


    def get_action(self, o, deterministic=False):
        if self.is_cuda:
            obs = torch.as_tensor(o, dtype=torch.float32).cuda()
        else:
            obs = torch.as_tensor(o, dtype=torch.float32)
        return self.ac.act(obs, deterministic)


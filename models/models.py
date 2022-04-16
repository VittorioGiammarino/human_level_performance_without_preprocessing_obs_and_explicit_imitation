#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 17:55:49 2021

@author: vittorio
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class SoftmaxActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SoftmaxActor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 128)
        nn.init.uniform_(self.l1.weight, -0.5, 0.5)
        self.l2 = nn.Linear(128,128)
        nn.init.uniform_(self.l2.weight, -0.5, 0.5)
        self.l3 = nn.Linear(128,action_dim)
        nn.init.uniform_(self.l3.weight, -0.5, 0.5)
        self.lS = nn.Softmax(dim=1)
        
        self.inverse_model_action = nn.Sequential(
            nn.Linear(2*state_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, action_dim)
            )
        
        self.inverse_model_reward = nn.Sequential(
            nn.Linear(2*state_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Sigmoid()
            )
        
        # Initialize parameters correctly
        self.apply(init_params)
        
    def forward(self, state):
        a = self.l1(state)
        a = F.relu(self.l2(a))
        return self.lS(self.l3(a))
    
    def sample(self, state):
        self.log_Soft = nn.LogSoftmax(dim=1)
        a = self.l1(state)
        a = F.relu(self.l2(a))
        log_prob = self.log_Soft(self.l3(a)) 
        
        prob = self.forward(state)
        m = Categorical(prob)
        action = m.sample()
        
        log_prob_sampled = log_prob.gather(1, action.reshape(-1,1).long())
        # log_prob_sampled = log_prob[torch.arange(len(action)),action]
        
        return action, log_prob_sampled
    
    def sample_log(self, state, action):
        self.log_Soft = nn.LogSoftmax(dim=1)
        a = self.l1(state)
        a = F.relu(self.l2(a))
        log_prob = self.log_Soft(torch.clamp(self.l3(a),-10,10)) 
                    
        log_prob_sampled = log_prob.gather(1, action.detach().reshape(-1,1).long())
        # log_prob_sampled = log_prob[torch.arange(len(action)), action]
        
        return log_prob, log_prob_sampled.reshape(-1,1)
    
    def forward_inv_a(self, embedding, embedding_next):
        hh = torch.cat((embedding, embedding_next), dim=-1)
        a = self.inverse_model_action(hh)
        
        return self.lS(torch.clamp(a,-10,10))
    
    def forward_inv_reward(self, embedding, embedding_next):
        hh = torch.cat((embedding, embedding_next), dim=-1)
        r = self.inverse_model_reward(hh)
        
        return r
        
    def sample_inverse_model(self, embedding, embedding_next):
        prob = self.forward_inv_a(embedding, embedding_next)
        m = Categorical(prob)
        action = m.sample()
        
        return action    
            
class Critic_flat_discrete(nn.Module):
    def __init__(self, state_dim, action_cardinality):
        super(Critic_flat_discrete, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_cardinality)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, action_cardinality)

    def forward(self, state):      
        q1 = F.relu(self.l1(state))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(state))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state):    
        q1 = F.relu(self.l1(state))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
    
class Value_net(nn.Module):
    def __init__(self, state_dim):
        super(Value_net, self).__init__()
        # Value_net architecture
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state):
        q1 = F.relu(self.l1(state))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)    
        return q1
    

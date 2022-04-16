#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 09:54:01 2021

@author: vittorio
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.models import SoftmaxActor
from models.models import Critic_flat_discrete
from models.models import Value_net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class on_off_AWAC_Q_lambda_Peng_obs:
    def __init__(self, state_dim, action_dim, action_space_cardinality, max_action, min_action, encoding_info = None, Entropy = True,   
                 num_steps_per_rollout=2000, intrinsic_reward = 0.01, number_obs_off_per_traj=1000, l_rate_actor=3e-4, l_rate_alpha=3e-4, 
                 discount=0.99, tau=0.005, beta=3, gae_gamma = 0.99, gae_lambda = 0.99, minibatch_size=64, num_epochs=10, alpha=0.2, critic_freq=2):
        

        self.actor = SoftmaxActor(state_dim, action_space_cardinality).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=l_rate_actor)
        
        self.critic = Critic_flat_discrete(state_dim, action_space_cardinality).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=l_rate_actor)
        
        self.critic_target = copy.deepcopy(self.critic)
        
        self.action_space = "Discrete"
            
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space_cardinality = action_space_cardinality
        self.max_action = max_action
        self.encoding_info = encoding_info
        
        self.num_steps_per_rollout_on = num_steps_per_rollout
        self.discount = discount
        self.tau = tau
        self.beta = beta
        self.critic_freq = critic_freq
        self.gae_gamma = gae_gamma
        self.gae_lambda = gae_lambda
        self.minibatch_size = minibatch_size
        self.num_epochs = num_epochs
        
        self.states = []
        self.actions = []
        self.target_Q = []
        self.advantage = []
        self.reward = []
        
        self.Entropy = Entropy
        self.target_entropy = -torch.FloatTensor([action_dim]).to(device)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr = l_rate_alpha) 
        self.alpha = alpha
        
        self.Total_t = 0
        self.Total_iter = 0
        self.total_it = 0
        
        self.number_obs_off_per_traj = int(number_obs_off_per_traj)
        self.intrinsic_reward = intrinsic_reward
        
    def reset_counters(self):
        self.Total_t = 0
        self.Total_iter = 0
        
    def encode_state(self, state):
        state = state.flatten()
        coordinates = state[0:2]
        psi = state[2]
        psi_encoded = np.zeros(self.encoding_info[0])
        psi_encoded[int(psi)]=1
        coin_dir_encoded = np.zeros(self.encoding_info[1])
        coin_dir = state[3]
        coin_dir_encoded[int(coin_dir)]=1
        current_state_encoded = np.concatenate((coordinates,psi_encoded,coin_dir_encoded))
        return current_state_encoded
    
    def encode_action(self, action):
        action_encoded = np.zeros(self.action_dim)
        action_encoded[int(action)]=1
        return action_encoded
                
    def select_action(self, state):
        state = torch.FloatTensor(self.encode_state(state).reshape(1,-1)).to(device)
        with torch.no_grad():
            if self.action_space == "Discrete":
                action, _ = self.actor.sample(state)
                return int((action).cpu().data.numpy().flatten())
            
            if self.action_space == "Continuous":
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                action, _, _ = self.actor.sample(state)
                return (action).cpu().data.numpy().flatten()
        
    def Q_lambda_Peng(self, env, args, reset = 'random', init_state = np.array([0,0,0,8])):
        step = 0
        self.Total_iter += 1
        self.states = []
        self.actions = []
        self.target_Q = []
        self.advantage = []
        self.reward = []

        while step < self.num_steps_per_rollout_on: 
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_gammas = []
            episode_lambdas = []    
            state, done = env.reset(reset, init_state), False
            t=0
            episode_reward = 0

            while not done and step < self.num_steps_per_rollout_on: 
                action = on_off_AWAC_Q_lambda_Peng_obs.select_action(self, state)
                
                state_encoded = self.encode_state(state.flatten())
            
                self.states.append(state_encoded)
                self.actions.append(action)
                episode_states.append(state_encoded)
                episode_actions.append(action)
                episode_gammas.append(self.gae_gamma**t)
                episode_lambdas.append(self.gae_lambda**t)
                
                state, reward, done, _ = env.step(action)
                
                episode_rewards.append(reward)
            
                t+=1
                step+=1
                episode_reward+=reward
                self.Total_t += 1
                        
            if done: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(f"Total T: {self.Total_t}, Iter Num: {self.Total_iter}, Episode T: {t} Reward: {episode_reward:.3f}")
                
            episode_states = torch.FloatTensor(np.array(episode_states)).to(device)
            
            if self.action_space == "Discrete":
                episode_actions = torch.LongTensor(np.array(episode_actions)).to(device)
            elif self.action_space == "Continuous":
                episode_actions = torch.FloatTensor(np.array(episode_actions))
            
            episode_rewards = torch.FloatTensor(np.array(episode_rewards)).to(device)
            episode_gammas = torch.FloatTensor(np.array(episode_gammas)).to(device)
            episode_lambdas = torch.FloatTensor(np.array(episode_lambdas)).to(device)
            
            traj_size = t
            
            self.actor.eval()
            self.critic.eval()
            
            with torch.no_grad():
                pi_action, log_pi = self.actor.sample(episode_states)
                Q1_on, Q2_on = self.critic_target(episode_states)
                
                if self.Entropy:
                    current_Q1 = Q1_on.gather(1, pi_action.long().unsqueeze(-1)) 
                    current_Q2 = Q2_on.gather(1, pi_action.long().unsqueeze(-1)) 
                    values = torch.min(current_Q1, current_Q2) - self.alpha*log_pi
                else:
                    current_Q1 = Q1_on.gather(1, pi_action.long().unsqueeze(-1)) 
                    current_Q2 = Q2_on.gather(1, pi_action.long().unsqueeze(-1)) 
                    values = torch.min(current_Q1, current_Q2)
                    
                final_bootstrap = values[-1].unsqueeze(-1)
                next_values = values[1:]
                next_action_values = torch.cat((episode_rewards[:-1].unsqueeze(-1) + self.gae_gamma*next_values, final_bootstrap))
                
                episode_adv = []
                episode_Q = []
                
                for j in range(traj_size):
                    off_policy_adjust = torch.cat((torch.FloatTensor([[0.]]).to(device), values[j+1:]))     
                    episode_deltas = next_action_values[j:] - off_policy_adjust
                    episode_Q.append(((episode_gammas*episode_lambdas)[:traj_size-j].unsqueeze(-1)*episode_deltas).sum())
                    episode_adv.append(((episode_gammas*episode_lambdas)[:traj_size-j].unsqueeze(-1)*episode_deltas).sum() - values[j])
                
                episode_advantage = torch.FloatTensor(episode_adv).to(device)
                episode_target_Q = torch.FloatTensor(episode_Q).to(device)
            
                self.advantage.append(episode_advantage)
                self.target_Q.append(episode_target_Q)
                self.reward.append(episode_rewards)
    
    def Q_lambda_Peng_off(self, off_policy_data, ntrajs):
        states = []
        actions = []
        target_Q = []
        advantage = []
        gammas_list = []
        lambdas_list = []
        
        size_off_policy_data = len(off_policy_data)
        ind = np.random.randint(0, size_off_policy_data-self.number_obs_off_per_traj-1, size=ntrajs)
        
        sampled_states = []
        sampled_actions = []
        sampled_rewards = []
        
        self.actor.eval()
        self.critic.eval()
        
        with torch.no_grad():
        
            for i in range(ntrajs):
                states_temp = torch.FloatTensor(off_policy_data[ind[i]:int(ind[i]+self.number_obs_off_per_traj)]).to(device)
                next_states_temp = torch.FloatTensor(off_policy_data[int(ind[i]+1):int(ind[i]+self.number_obs_off_per_traj+1)]).to(device)
                actions_temp = self.actor.sample_inverse_model(states_temp, next_states_temp)
                rewards_temp = self.actor.forward_inv_reward(states_temp, next_states_temp)
                rewards_i = self.intrinsic_reward*torch.ones_like(rewards_temp)    
                rewards_tot = rewards_temp + rewards_i
            
                sampled_states.append(states_temp)
                sampled_actions.append(actions_temp)
                sampled_rewards.append(rewards_tot)
            
            for l in range(ntrajs):
                traj_size = self.number_obs_off_per_traj
                gammas = []
                lambdas = []
                for t in range(traj_size):
                    gammas.append(self.gae_gamma**t)
                    lambdas.append(self.gae_lambda**t)
                    
                gammas_list.append(torch.FloatTensor(np.array(gammas)).to(device))
                lambdas_list.append(torch.FloatTensor(np.array(lambdas)).to(device))
                    
            for l in range(ntrajs):
                
                episode_states = sampled_states[l]
                episode_actions = sampled_actions[l]
                episode_rewards = sampled_rewards[l] 
                episode_gammas = gammas_list[l]
                episode_lambdas = lambdas_list[l]
                
                traj_size = self.number_obs_off_per_traj
                
                pi_action, log_pi = self.actor.sample(episode_states)
                Q1_on, Q2_on = self.critic_target(episode_states)
                
                if self.Entropy:
                    current_Q1 = Q1_on.gather(1, pi_action.long().unsqueeze(-1)) 
                    current_Q2 = Q2_on.gather(1, pi_action.long().unsqueeze(-1)) 
                    values = torch.min(current_Q1, current_Q2) - self.alpha*log_pi
                else:
                    current_Q1 = Q1_on.gather(1, pi_action.long().unsqueeze(-1)) 
                    current_Q2 = Q2_on.gather(1, pi_action.long().unsqueeze(-1)) 
                    values = torch.min(current_Q1, current_Q2)
                    
                final_bootstrap = values[-1].unsqueeze(-1)
                next_values = values[1:]
                next_action_values = torch.cat((episode_rewards[:-1] + self.gae_gamma*next_values, final_bootstrap))
                
                episode_adv = []
                episode_Q = []
                
                for j in range(traj_size):
                    off_policy_adjust = torch.cat((torch.FloatTensor([[0.]]).to(device), values[j+1:]))     
                    episode_deltas = next_action_values[j:] - off_policy_adjust
                    episode_Q.append(((episode_gammas*episode_lambdas)[:traj_size-j].unsqueeze(-1)*episode_deltas).sum())
                    episode_adv.append(((episode_gammas*episode_lambdas)[:traj_size-j].unsqueeze(-1)*episode_deltas).sum() - values[j])
                
                episode_advantage = torch.FloatTensor(episode_adv).to(device)
                episode_target_Q = torch.FloatTensor(episode_Q).to(device)
            
                states.append(episode_states)
                actions.append(episode_actions)
                target_Q.append(episode_target_Q)
                advantage.append(episode_advantage)
                    
        return states, actions, target_Q, advantage
    
    def train_inverse_models(self):
        states_on = torch.FloatTensor(np.array(self.states)).to(device)
        
        if self.action_space == "Discrete":
            actions_on = torch.LongTensor(np.array(self.actions)).to(device)
        elif self.action_space == "Continuous":
            actions_on = torch.FloatTensor(np.array(self.actions)).to(device)
        
        reward_on = torch.cat(self.reward)
        
        max_steps = self.num_epochs * (self.num_steps_per_rollout_on // self.minibatch_size)
        
        for _ in range(max_steps):
            
            minibatch_indices_ims = np.random.choice(range(self.num_steps_per_rollout_on-1), self.minibatch_size, False)
            states_ims = states_on[minibatch_indices_ims]
            next_states_ims = states_on[minibatch_indices_ims+1]
            rewards_ims = reward_on[minibatch_indices_ims]
            actions_ims = actions_on[minibatch_indices_ims]
            
            inverse_action_model_prob = self.actor.forward_inv_a(states_ims, next_states_ims)
            m = F.one_hot(actions_ims.squeeze().cpu(), self.action_space_cardinality).float().to(device)
            L_ia = F.mse_loss(inverse_action_model_prob, m)
            
            L_ir = F.mse_loss(rewards_ims.unsqueeze(-1), self.actor.forward_inv_reward(states_ims, next_states_ims))
              
            self.actor_optimizer.zero_grad()
            loss = L_ia + L_ir 
            loss.backward()
            self.actor_optimizer.step()
    
    def train(self, states_off, actions_off, target_Q_off, advantage_off):
        self.total_it += 1
        
        states_on = torch.FloatTensor(np.array(self.states)).to(device)
        
        if self.action_space == "Discrete":
            actions_on = torch.LongTensor(np.array(self.actions)).to(device)
        elif self.action_space == "Continuous":
            actions_on = torch.FloatTensor(np.array(self.actions)).to(device)
                
        target_Q_on = torch.cat(self.target_Q)
        advantage_on = torch.cat(self.advantage).to(device)
        
        states_off = torch.cat(states_off)
        rollout_states = torch.cat((states_on, states_off))
        
        actions_off = torch.cat(actions_off)
        rollout_actions = torch.cat((actions_on, actions_off.squeeze()))
        
        target_Q_off = torch.cat(target_Q_off)
        rollout_target_Q = torch.cat((target_Q_on, target_Q_off))
        
        advantage_off = torch.cat(advantage_off)
        rollout_advantage = torch.cat((advantage_on, advantage_off))
        
        rollout_advantage = (rollout_advantage-rollout_advantage.mean())/(rollout_advantage.std()+1e-6)
        
        self.actor.train()
        self.critic.train()
        self.num_steps_per_rollout = len(rollout_advantage)
        max_steps = self.num_epochs * (self.num_steps_per_rollout // self.minibatch_size)
        
        for i in range(max_steps):
            
            minibatch_indices = np.random.choice(range(self.num_steps_per_rollout), self.minibatch_size, False)
            batch_actions = rollout_actions[minibatch_indices]
            batch_target_Q = rollout_target_Q[minibatch_indices]
            batch_advantage = rollout_advantage[minibatch_indices]
            batch_states = rollout_states[minibatch_indices]
                    
            if self.action_space == "Discrete":
                log_prob, log_prob_rollout = self.actor.sample_log(batch_states, batch_actions)

            elif self.action_space == "Continuous": 
                log_prob_rollout = self.actor.sample_log(batch_states, batch_actions)
                
            r = (log_prob_rollout).squeeze()
            weights = F.softmax(batch_advantage/self.beta, dim=0).detach()
            L_clip = r*weights
            
            if self.action_space == "Discrete":
                Q1, Q2 = self.critic(batch_states)
                current_Q1 = Q1.gather(1, batch_actions.long().unsqueeze(-1)) 
                current_Q2 = Q2.gather(1, batch_actions.long().unsqueeze(-1)) 
            
            elif self.action_space == "Continuous":
                #current Q estimates
                current_Q1, current_Q2 = self.critic(batch_states, batch_actions)
            
            critic_loss = F.mse_loss(current_Q1.squeeze(), batch_target_Q) + F.mse_loss(current_Q1.squeeze(), batch_target_Q)
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            if self.Entropy:
                _, log_pi_state = self.actor.sample(batch_states)
                loss = (-1) * (L_clip - self.alpha*log_pi_state).mean() + critic_loss 
            else:
                loss = (-1) * (L_clip).mean() + critic_loss 
            
            loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
            if self.Entropy: 

                alpha_loss = -(self.log_alpha * (log_pi_state + self.target_entropy).detach()).mean()
        
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
        
                self.alpha = self.log_alpha.exp()
                
            # Update the frozen target models
            if self.total_it % self.critic_freq == 0:
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
    def save_actor(self, filename):
        option = 0
        torch.save(self.actor.state_dict(), filename + f"_pi_lo_option_{option}")
        torch.save(self.actor_optimizer.state_dict(), filename + f"_pi_lo_optimizer_option_{option}")
    
    def load_actor(self, filename):
        option = 0
        self.actor.load_state_dict(torch.load(filename + f"_pi_lo_option_{option}"))
        self.actor_optimizer.load_state_dict(torch.load(filename + f"_pi_lo_optimizer_option_{option}"))

    def save_critic(self, filename):
        torch.save(self.value_function.state_dict(), filename + "_value_function")
        torch.save(self.value_function_optimizer.state_dict(), filename + "_value_function_optimizer")
    
    def load_critic(self, filename):
        self.value_function.load_state_dict(torch.load(filename + "_value_function"))      
        self.value_function_optimizer.load_state_dict(torch.load(filename + "_value_function_optimizer")) 
        

        

        
        
        
        
        
        

            
            
        
            
            
            

        

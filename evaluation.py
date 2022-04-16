#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 18:02:25 2021

@author: vittorio
"""

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def FlatStochasticSampleTrajMDP(policy, env, args, reset = 'random', initial_state = np.array([0,0,0,8])):

    Reward_array = np.empty((0,0),int)
   
    policy.actor.eval()
    
    for t in range(args.evaluation_episodes):
        state, done = env.reset(reset, initial_state), False
        
        size_input = len(state)
        x = np.empty((0, size_input))
        x = np.append(x, state.reshape(1, size_input), 0)
        u_tot = np.empty((0,0),int)
        cum_reward = 0 
        
        for _ in range(0, args.evaluation_max_n_steps):
            action = policy.select_action(np.array(state))
            u_tot = np.append(u_tot, action) 
            
            state, reward, done, _ = env.step(action)
            x = np.append(x, state.reshape(1, size_input), 0)
            cum_reward = cum_reward + reward  
            
        Reward_array = np.append(Reward_array, cum_reward)
        
    return Reward_array  
            
def eval_policy(seed, policy, env, args, reset = 'random', initial_state = np.array([0,0,0,8])):

    Reward = FlatStochasticSampleTrajMDP(policy, env, args, reset, initial_state)
    avg_reward = np.sum(Reward)/args.evaluation_episodes

    print("---------------------------------------")
    print(f"Seed {seed}, Evaluation over {args.evaluation_episodes} episodes, reward: {avg_reward:.3f}")
    print("---------------------------------------")
    
    return avg_reward

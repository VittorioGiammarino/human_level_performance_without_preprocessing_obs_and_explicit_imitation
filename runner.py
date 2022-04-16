#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 10:42:55 2021

@author: vittorio
"""

import time
from evaluation import eval_policy
    
#################################################################################
# on-off RL
#################################################################################
    
class run_on_off_AWAC_Q_lambda_Peng:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, args, seed):
                # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(seed, self.agent, env, args)
        evaluation_RL.append(avg_reward) 
        start_time = time.time()
    
        for i in range(int(args.max_iter)):
                        
            self.agent.Q_lambda_Peng(env, args)
            
            if args.mode == "on_off_RL_from_observations":
                self.agent.train_inverse_models()
            
            states, actions, target_Q, advantage = self.agent.Q_lambda_Peng_off(replay_buffer, args.ntrajs)
            self.agent.train(states, actions, target_Q, advantage)

            # Evaluate episode
            if (i + 1) % args.eval_freq == 0:
                avg_reward = eval_policy(seed, self.agent, env, args)
                evaluation_RL.append(avg_reward)  
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent 
    
class run_on_off_AWAC_Q_lambda_Haru:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, args, seed):
                # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(seed, self.agent, env, args)
        evaluation_RL.append(avg_reward) 
        start_time = time.time()
    
        for i in range(int(args.max_iter)):
                        
            self.agent.Q_lambda_Haru(env, args)
            
            if args.mode == "on_off_RL_from_observations":
                self.agent.train_inverse_models()
            
            states, actions, target_Q, advantage = self.agent.Q_lambda_Haru_off(replay_buffer, args.ntrajs)
            self.agent.train(states, actions, target_Q, advantage)

            # Evaluate episode
            if (i + 1) % args.eval_freq == 0:
                avg_reward = eval_policy(seed, self.agent, env, args)
                evaluation_RL.append(avg_reward)  
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent 
    
class run_on_off_AWAC_TB:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, args, seed):
                # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(seed, self.agent, env, args)
        evaluation_RL.append(avg_reward) 
        start_time = time.time()
    
        for i in range(int(args.max_iter)):
                        
            self.agent.TB_lambda(env, args)
            
            if args.mode == "on_off_RL_from_observations":
                self.agent.train_inverse_models()
            
            states, actions, target_Q, advantage = self.agent.TB_lambda_off(replay_buffer, args.ntrajs)
            self.agent.train(states, actions, target_Q, advantage)

            # Evaluate episode
            if (i + 1) % args.eval_freq == 0:
                avg_reward = eval_policy(seed, self.agent, env, args)
                evaluation_RL.append(avg_reward)  
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent 
    
class run_on_off_AWAC_GAE:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, replay_buffer, args, seed):
                # Evaluate untrained policy
        evaluation_RL = []
        wallclock_time = []
        avg_reward = eval_policy(seed, self.agent, env, args)
        evaluation_RL.append(avg_reward) 
        start_time = time.time()
    
        for i in range(int(args.max_iter)):
                        
            self.agent.GAE(env, args)
            
            if args.mode == "on_off_RL_from_observations":
                self.agent.train_inverse_models()
            
            states, actions, returns, advantage = self.agent.GAE_off(replay_buffer, args.ntrajs)
            self.agent.train(states, actions, returns, advantage)

            # Evaluate episode
            if (i + 1) % args.eval_freq == 0:
                avg_reward = eval_policy(seed, self.agent, env, args)
                evaluation_RL.append(avg_reward)  
                wallclock_time.append(time.time()-start_time)
                 
        return wallclock_time, evaluation_RL, self.agent 

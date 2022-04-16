#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:24:05 2020

@author: vittorio
"""
import torch
import argparse
import os
import numpy as np
import pickle

import World
from utils import Encode_Data
import runner

from algorithms.on_off_AWAC_Q_lambda_Peng_obs import on_off_AWAC_Q_lambda_Peng_obs
from algorithms.on_off_AWAC_Q_lambda_Haru_obs import on_off_AWAC_Q_lambda_Haru_obs
from algorithms.on_off_AWAC_TB_obs import on_off_AWAC_TB_obs
from algorithms.on_off_AWAC_GAE_obs import on_off_AWAC_GAE_obs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%

def RL(env, args, seed):
    if args.action_space == 'Continuous':
        action_dim = env.action_space.shape[0] 
        action_space_cardinality = np.inf
        max_action = np.zeros((action_dim,))
        min_action = np.zeros((action_dim,))
        for a in range(action_dim):
            max_action[a] = env.action_space.high[a]   
            min_action[a] = env.action_space.low[a]  
            
    elif args.action_space == 'Discrete':
        try:
            action_dim = env.action_space.shape[0] 
        except:
            action_dim = 1

        action_space_cardinality = env.action_size
        max_action = np.nan
        min_action = np.nan
    
    Trajectories = np.load("./Expert_data/Trajectories.npy", allow_pickle=True).tolist() 
    TrainingSet = Trajectories[args.human*10:args.human*10+10]
    
    TrainingSet_human = np.empty((0,4)) 
    for i in range(10):
        TrainingSet_human = np.concatenate((TrainingSet_human, TrainingSet[i]))
        
    off_policy_observations, encoding_info = Encode_Data(TrainingSet_human)
    state_dim = off_policy_observations.shape[1]
                 
    if args.policy == "AWAC_Q_lambda_Peng":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "action_space_cardinality": action_space_cardinality,
         "max_action": max_action,
         "min_action": min_action,
         "encoding_info": encoding_info,
         "Entropy": args.Entropy,
         "num_steps_per_rollout": args.number_steps_per_iter,
         "intrinsic_reward": args.intrinsic_reward
        }

        Agent_RL = on_off_AWAC_Q_lambda_Peng_obs(**kwargs)
        
        run_sim = runner.run_on_off_AWAC_Q_lambda_Peng(Agent_RL)
        wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, off_policy_observations, args, seed)
        
        return wallclock_time, evaluation_RL, Agent_RL 
    
    if args.policy == "AWAC_Q_lambda_Haru":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "action_space_cardinality": action_space_cardinality,
         "max_action": max_action,
         "min_action": min_action,
         "encoding_info": encoding_info,
         "Entropy": args.Entropy,
         "num_steps_per_rollout": args.number_steps_per_iter,
         "intrinsic_reward": args.intrinsic_reward
        }

        Agent_RL = on_off_AWAC_Q_lambda_Haru_obs(**kwargs)
        
        run_sim = runner.run_on_off_AWAC_Q_lambda_Haru(Agent_RL)
        wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, off_policy_observations, args, seed)
        
        return wallclock_time, evaluation_RL, Agent_RL  
    
    if args.policy == "AWAC_Q_lambda_TB":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "action_space_cardinality": action_space_cardinality,
         "max_action": max_action,
         "min_action": min_action,
         "encoding_info": encoding_info,
         "Entropy": args.Entropy,
         "num_steps_per_rollout": args.number_steps_per_iter,
         "intrinsic_reward": args.intrinsic_reward
        }

        Agent_RL = on_off_AWAC_TB_obs(**kwargs)
        
        run_sim = runner.run_on_off_AWAC_TB(Agent_RL)
        wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, off_policy_observations, args, seed)
        
        return wallclock_time, evaluation_RL, Agent_RL  
    
    if args.policy == "AWAC_GAE":        
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "action_space_cardinality": action_space_cardinality,
         "max_action": max_action,
         "min_action": min_action,
         "encoding_info": encoding_info,
         "Entropy": args.Entropy,
         "num_steps_per_rollout": args.number_steps_per_iter,
         "intrinsic_reward": args.intrinsic_reward
        }

        Agent_RL = on_off_AWAC_GAE_obs(**kwargs)
        
        run_sim = runner.run_on_off_AWAC_GAE(Agent_RL)
        wallclock_time, evaluation_RL, Agent_RL = run_sim.run(env, off_policy_observations, args, seed)
        
        return wallclock_time, evaluation_RL, Agent_RL  
    
def train(env, args, seed): 
    
    # Set seeds
    env.Seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
            
    wallclock_time, evaluations, policy = RL(env, args, seed)
    
    return wallclock_time, evaluations, policy


if __name__ == "__main__":
    
    Trajectories = np.load("./Expert_data/Trajectories.npy", allow_pickle=True).tolist()
    Coins_location = np.load("./Expert_data/Coins_location.npy")
    
    len_trajs = []
    for i in range(len(Trajectories)):
        len_trajs.append(len(Trajectories[i]))
        
    mean_len_trajs = int(np.mean(len_trajs))
    
    parser = argparse.ArgumentParser()
    #General
    parser.add_argument("--mode", default="on_off_RL_from_observations")     # number of options
    parser.add_argument("--policy", default="AWAC_Q_lambda_TB")                   # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--seed", default=10, type=int)               # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--env", default="Foraging")               # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--action_space", default="Discrete")
    parser.add_argument("--number_steps_per_iter", default=30000, type=int) # Time steps initial random policy is used 25e3
    parser.add_argument("--eval_freq", default=1, type=int)          # How often (time steps) we evaluate
    parser.add_argument("--max_iter", default=334, type=int)    # Max time steps to run environment
    parser.add_argument("--human", default=0, type=int)
    # HRL
    parser.add_argument("--start_timesteps", default=25e3, type=int) #Time steps before training default=25e3
    parser.add_argument("--expl_noise", default=0.1)   
    parser.add_argument("--save_model", action="store_false")         #Save model and optimizer parameters
    parser.add_argument("--load_model", default=True, type=bool)              #Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--load_model_path", default="") 
    # off-policy
    parser.add_argument("--ntrajs", default=10, type=int)
    parser.add_argument("--intrinsic_reward", default=0.005, type=float)
    parser.add_argument("--Entropy", action="store_true")
    # Evaluation
    parser.add_argument("--evaluation_episodes", default=10, type=int)
    parser.add_argument("--evaluation_max_n_steps", default = mean_len_trajs, type=int)
    # Experiments
    parser.add_argument("--adv_reward", action="store_true") 
    
    args = parser.parse_args()
        
    if args.mode == "on_off_RL_from_observations":
        
        assert args.human >= 0 and args.human <= 4
    
        file_name = f"{args.mode}_{args.policy}_human{args.human}_ri_{args.intrinsic_reward}_{args.seed}"
        print("---------------------------------------")
        print(f"Policy: {args.policy}, Human: {args.human}, ri: {args.intrinsic_reward}, Env: {args.env}, Seed: {args.seed}")
        print("---------------------------------------")
               
        if not os.path.exists(f"./results/{args.mode}"):
            os.makedirs(f"./results/{args.mode}")
            
        if not os.path.exists(f"./Saved_models/{args.mode}/{file_name}"):
            os.makedirs(f"./Saved_models/{args.mode}/{file_name}")
         
        coins_distribution = args.human*10    
        coins_location = Coins_location[coins_distribution,:,:] 
        env = World.Foraging.env(coins_location)
    
        evaluations, policy = train(env, args, args.seed)
        
        if args.save_model: 
            np.save(f"./results/{args.mode}/evaluation_{file_name}", evaluations)
            policy.save_actor(f"./Saved_models/{args.mode}/{file_name}/{file_name}")
            policy.save_critic(f"./Saved_models/{args.mode}/{file_name}/{file_name}")
            

                
                
                
                
        
        
        
        
        
                
    


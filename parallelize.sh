#!/bin/bash

for seed in $(seq 0 9);
do
for expert_traj in $(seq 0 4);
do 
qsub QSUBS/intrinsic_reward_0.01/AWAC_GAE.qsub $seed $expert_traj
qsub QSUBS/intrinsic_reward_0.01/AWAC_Q_lambda_Haru.qsub $seed $expert_traj
qsub QSUBS/intrinsic_reward_0.01/AWAC_Q_lambda_Peng.qsub $seed $expert_traj
qsub QSUBS/intrinsic_reward_0.01/AWAC_Q_lambda_TB.qsub $seed $expert_traj
qsub QSUBS/intrinsic_reward_0.005/AWAC_GAE.qsub $seed $expert_traj
qsub QSUBS/intrinsic_reward_0.005/AWAC_Q_lambda_Haru.qsub $seed $expert_traj
qsub QSUBS/intrinsic_reward_0.005/AWAC_Q_lambda_Peng.qsub $seed $expert_traj
qsub QSUBS/intrinsic_reward_0.005/AWAC_Q_lambda_TB.qsub $seed $expert_traj
done 
done

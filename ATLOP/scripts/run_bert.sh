#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --partition=edu1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --job-name=DRE
#SBATCH -o /home2/csh102/DocRE/slurm/out/%j_%N.out  # STDOUT 
#SBATCH -e /home2/csh102/DocRE/slurm/err/%j_%N.err  # STDERR

echo "start at:" `date` 
echo "node: $HOSTNAME" 
echo "jobid: $SLURM_JOB_ID" 
echo "--------------------------------------"

module unload CUDA/11.2.2 
module load cuda/11.3.1

/home2/csh102/anaconda3/envs/240504-gnn-survey/bin/python /home2/csh102/DocRE/ATLOP/code/train.py \
    --data_dir /home2/csh102/DocRE/docred \
    --train_file train_annotated.json \
    --dev_file dev_env.json \
    --test_file dev_env.json \
    \
    --transformer_type bert \
    --model_name_or_path bert-base-cased \
    --train_batch_size 4 \
    --test_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_labels 4 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.06 \
    --num_train_epochs 30.0 \
    --seed 66 \
    --num_class 97
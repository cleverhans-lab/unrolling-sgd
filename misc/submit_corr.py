#!/usr/bin/env python
import time
import os
import itertools

r_list= [0]
r_list = [0.0,0.1,1.0,5.0,10.0,15.0,20.0,25.0,50.0,100.0,150.0,200.0,250.0]#[0.0,0.01,0.1,1.0,2.0,5.0,10.0,15.0,20.0]
l2_regs = [0]#[0.0,0.00001,0.0001,0.001,0.01,0.1,0.5,1.0,3.0,5.0]
ee_list =[200]
model = ['resnet','vgg']
loss_func = ['std']
dataset = ['cifar10','cifar100']
pretrain_epochs = [20,70]
pretrain_batch_size = [128]
finetune_batch_size = [32]
finetune_epochs = [5]

big_list = [l2_regs,ee_list,model,loss_func,dataset,pretrain_epochs,pretrain_batch_size,finetune_batch_size, finetune_epochs,r_list]

combinations = list(itertools.product(*big_list))

print('number of jobs', len(combinations))

for instance in combinations:
    l2_r,ee,m,l,d,pe,pb,fb,fe,r= instance
    job_file = f"./job_files/corr_{m}_{d}_{l}.slrm"
    with open(job_file, 'w+') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --time=50:00:00\n")
        fh.writelines("#SBATCH --mem=20g\n")
        fh.writelines("#SBATCH --cpus-per-task=2\n")
        fh.writelines("#SBATCH --gres=gpu:1\n")
        fh.writelines("#SBATCH --partition=t4v2\n")
        fh.writelines("#SBATCH --qos=high\n")
        fh.writelines("#SBATCH --open-mode=append\n")
        fh.writelines("echo `date`: Job $SLURM_JOB_ID is allocated resource\n")
        fh.writelines("ln -sfn /checkpoint/${USER}/${SLURM_JOB_ID} $PWD/models\n")
        fh.writelines("touch /checkpoint/${USER}/${SLURM_JOB_ID}/DELAYPURGE\n")
        fh.writelines("export LD_LIBRARY_PATH=/pkgs/cuda-10.1/lib64:/pkgs/cudnn-10.1-v7.6.3.30/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}\n")
        fh.writelines("export PATH=/pkgs/anaconda3/bin:$PATH\n")
        fh.writelines("source activate /scratch/ssd001/home/$USER/learning/\n")
        fh.writelines(f"python -u v2_fixed_correlation.py --l2_regularizer {l2_r} --model {m} --loss_func {l} --dataset {d} --pretrain_epochs {pe} --pretrain_batch_size {pb} --finetune_batch_size {fb} --finetune_epochs {fe} --regularizer {r} --eval_every {ee}\n")
        fh.writelines(f"conda deactivate\n")
    os.system("sbatch %s" %job_file)
    time.sleep(0.3)

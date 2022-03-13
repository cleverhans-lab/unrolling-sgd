#!/usr/bin/env python
import time
import os
import itertools

models = ['vgg','resnet']
loss_func = ['regular','std']
dataset = ['cifar10','cifar100']
pretrain_epochs = [50]
pretrain_batch_size = [128]
finetune_batch_size = [32]
finetune_epochs = [1]
checkpoint_freq_list = [10]

big_list = [models,loss_func,dataset,pretrain_epochs,pretrain_batch_size,finetune_batch_size,finetune_epochs,checkpoint_freq_list]

combinations = list(itertools.product(*big_list))

print('number of jobs', len(combinations))

for instance in combinations:
    m,l,d,pe,pb,fb,fe,cf = instance
    job_file = f"./job_files/chk_{m}_{d}_{l}.slrm"
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
        fh.writelines(f"python -u checkpoint_version.py --model {m} --dataset {d} --loss_func {l} --pretrain_epochs {pe} --pretrain_batch_size {pb} --finetune_batch_size {fb} --finetune_epochs {fe} --checkpoint_freq {cf}\n")
        fh.writelines(f"conda deactivate\n")
    os.system("sbatch %s" %job_file)
    time.sleep(0.3)

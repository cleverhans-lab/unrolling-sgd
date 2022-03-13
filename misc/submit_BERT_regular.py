#!/usr/bin/env python
import time
import os
import itertools


lr_list=[2e-5,1e-5,5e-6,2e-6,1e-6]
model = ['linear']
pretrain_epochs = [200]
pretrain_batch_size = [8,16,32,64]
wd_list = [0.0,1e-5,1e-4,1e-3,1e-2,1e-1]
big_list = [lr_list,wd_list,model,pretrain_epochs,pretrain_batch_size]
combinations = list(itertools.product(*big_list))

print('number of jobs', len(combinations))

for i,instance in enumerate(combinations):
    lr,wd,m,pe,pb= instance
    job_file = f"./job_files/test_{i}.slrm"
    with open(job_file, 'w+') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --time=10:00:00\n")
        fh.writelines("#SBATCH --mem=50g\n")
        fh.writelines("#SBATCH --exclude=gpu090,gpu110,gpu072,gpu109\n")
        fh.writelines("#SBATCH --cpus-per-task=2\n")
        fh.writelines("#SBATCH --gres=gpu:2\n")
        fh.writelines("#SBATCH --partition=t4v2\n")
        fh.writelines("#SBATCH --qos=high\n")
        fh.writelines("#SBATCH --open-mode=append\n")
        fh.writelines("echo `date`: Job $SLURM_JOB_ID is allocated resource\n")
        fh.writelines("ln -sfn /checkpoint/${USER}/${SLURM_JOB_ID} $PWD/models\n")
        fh.writelines("touch /checkpoint/${USER}/${SLURM_JOB_ID}/DELAYPURGE\n")
        #h.writelines("export LD_LIBRARY_PATH=/pkgs/cuda-10.1/lib64:/pkgs/cudnn-10.1-v7.6.3.30/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}\n")
        #fh.writelines("export PATH=/pkgs/anaconda3/bin:$PATH\n")
        #fh.writelines("source activate /scratch/ssd001/home/$USER/learning/\n")
        fh.writelines(f"python -u regular_bert.py --weight_decay {wd} --lr {lr} --model {m} --pretrain_epochs {pe} --pretrain_batch_size {pb} \n")
        #fh.writelines(f"conda deactivate\n")
    os.system("sbatch %s" %job_file)


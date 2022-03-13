#!/usr/bin/env python
import time
import os
import itertools

r_list = [0,0.01,0.1,1.0,2.0,5.0,10.0,15.0,20.0,30.0,50.0]
lr_list = [2e-5]
model = ['linear']
pretrain_epochs = [200]
pretrain_batch_size = [8]
wd_list = [0.0]#1e-5,1e-4,1e-3,1e-2,1e-1]
big_list = [lr_list,wd_list,model,pretrain_epochs,pretrain_batch_size,r_list]
combinations = list(itertools.product(*big_list))

print('number of jobs', len(combinations))

for instance in combinations:
    lr,wd,m,pe,pb,r= instance
    job_file = f"./job_files/test_{lr}_{wd}_{r}_{m}_{pe}_{pb}.slrm"
    with open(job_file, 'w+') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --time=50:00:00\n")
        fh.writelines("#SBATCH --mem=60g\n")
        fh.writelines("#SBATCH --cpus-per-task=4\n")
        fh.writelines("#SBATCH --exclude=gpu090,gpu110,gpu072,gpu109,gpu064,gpu077\n")
        fh.writelines("#SBATCH --gres=gpu:2\n")
        fh.writelines("#SBATCH --partition=t4v2\n")
        fh.writelines("#SBATCH --qos=high\n")
        fh.writelines("#SBATCH --open-mode=append\n")
        fh.writelines("echo `date`: Job $SLURM_JOB_ID is allocated resource\n")
        fh.writelines("ln -sfn /checkpoint/${USER}/${SLURM_JOB_ID} $PWD/models\n")
        fh.writelines("touch /checkpoint/${USER}/${SLURM_JOB_ID}/DELAYPURGE\n")
        #fh.writelines("export LD_LIBRARY_PATH=/pkgs/cuda-10.1/lib64:/pkgs/cudnn-10.1-v7.6.3.30/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}\n")
        #fh.writelines("export PATH=/pkgs/anaconda3/bin:$PATH\n")
        #fh.writelines("source activate /scratch/ssd001/home/$USER/learning/\n")
        fh.writelines(f"python -u no_bert_training_unlearning.py --weight_decay {wd} --lr {lr} --model {m} --pretrain_epochs {pe} --pretrain_batch_size {pb} --regularizer {r}\n")
        #fh.writelines(f"conda deactivate\n")
    os.system("sbatch %s" %job_file)
    time.sleep(0.3)

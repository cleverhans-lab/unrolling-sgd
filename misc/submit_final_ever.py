#!/usr/bin/env python
import time
import os
import itertools

r_list = [0.0,5.0]
ee_list =[400]
decay = [1.0,2.0]
lr_list=[0.1]
model = ['resnet','vgg']
dataset = ['cifar10']
pretrain_epochs = [50]
pretrain_batch_size = [128]

big_list = [lr_list,ee_list,model,dataset,pretrain_epochs,pretrain_batch_size,r_list,decay]

combinations = list(itertools.product(*big_list))

print('number of jobs', len(combinations))

for instance in combinations:
    lr,ee,m,d,pe,pb,r,de= instance
    job_file = f"./job_files/corr_{m}_{d}.slrm"
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
        fh.writelines(f"python -u final_ever_file.py --lr {lr} --model {m} --dataset {d} --pretrain_epochs {pe} --pretrain_batch_size {pb} --regularizer {r} --eval_every {ee} --decay {de}\n")
        fh.writelines(f"conda deactivate\n")
    os.system("sbatch %s" %job_file)
    time.sleep(0.3)

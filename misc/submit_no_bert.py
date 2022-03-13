#!/usr/bin/env python
import time
import os
import itertools

job_file = f"./job_files/test.slrm"
with open(job_file, 'w+') as fh:
    fh.writelines("#!/bin/bash\n")
    fh.writelines("#SBATCH --time=10:00:00\n")
    fh.writelines("#SBATCH --mem=50g\n")
    fh.writelines("#SBATCH --cpus-per-task=2\n")
    fh.writelines("#SBATCH --gres=gpu:4\n")
    fh.writelines("#SBATCH --partition=t4v2\n")
    fh.writelines("#SBATCH --qos=high\n")
    fh.writelines("#SBATCH --open-mode=append\n")
    fh.writelines("echo `date`: Job $SLURM_JOB_ID is allocated resource\n")
    fh.writelines("ln -sfn /checkpoint/${USER}/${SLURM_JOB_ID} $PWD/models\n")
    fh.writelines("touch /checkpoint/${USER}/${SLURM_JOB_ID}/DELAYPURGE\n")
    #h.writelines("export LD_LIBRARY_PATH=/pkgs/cuda-10.1/lib64:/pkgs/cudnn-10.1-v7.6.3.30/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}\n")
    #fh.writelines("export PATH=/pkgs/anaconda3/bin:$PATH\n")
    #fh.writelines("source activate /scratch/ssd001/home/$USER/learning/\n")
    fh.writelines(f"python -u no_bert_training.py\n")
    #fh.writelines(f"conda deactivate\n")
os.system("sbatch %s" %job_file)

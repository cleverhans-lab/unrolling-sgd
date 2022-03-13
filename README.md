# Unlearning-Unrolling-SGD

Included is the relevant code for our work. The files are divided into their relevant experiments: Unlearning in the image domain (/Unlearning), training and unlearning BERT (/BERT), comparing Membership Inference to unlearning (/Membership inference) and Privacy Risk Score experiments (/Privacy_risk_score). Model definition files for resnet and vgg are in their appropriately named files. The miscellanous folder includes scripts on how to run each file.

## Setup
For all unlearning experiments, we are required to calculate singular values of the Hessian matrix of our model. To do so, we leverage the PyHessian library. We recommend cloning the repository from source as we have run into issues making it compatible with our setup if installed via pip or brew.


## Unlearning:
Master_unlearning.py trains an image model (Resnet or VGG based models) on a specific dataset (CIFAR-10/100) while measuring the unlearning error and relevant terms throughout the training procedure (i.e., singular values and change in weights). Once completed, relevant information during training is pickled. Check command line arguments to modify the training setup.

## Privacy Risk Score:
The first step to calculate privacy risk score is to train a regular model and the equivalent shadow model on half the dataset for each setting of interest (modified by the command line arguments) (in train_shadow_model.py). Calculate_PRS.py uses these models to calculate the distribution/bins for the unlearned data point. Lastly, privacy_risk_score.py uses these bins to calculate the privacy risk score for each setting. Correlation_PRS.py does the same except measures it for both the naively retrained model, the unlearned model and the original model. prs_vs_ver.py measures verification error in addition to the PRS.

## BERT:
For BERT, we finetune and unlearn BERT in no_bert_training_unlearning.py. Relevant statistics for unlearning are pickled and settings are modified by the command line arguments.

# Unlearning-Unrolling-SGD

Included is the relevant code for our work. The files are divided into their relevant experiments: Unlearning in the image domain (/Unlearning), training and unlearning BERT (/BERT), comparing Membership Inference to unlearning (/Membership inference) and Privacy Risk Score experiments (/Privacy_risk_score). Model definition files for resnet and vgg are in their appropriately named files. The miscellanous folder includes scripts on how to run each file.


## Unlearning:
Master_unlearning.py trains an image model (Resnet or VGG based models) on a specific dataset (CIFAR-10/100) while measuring the unlearning error and relevant terms throughout the training procedure (i.e., singular values and change in weights). Once completed, relevant information during training is pickled. Check command line arguments to modify the training setup.

# Calibrated Predictions with Covariate Shift via Unsupervised Domain Adaptation (AISTATS20)

This code reproduces the results in the [paper](http://proceedings.mlr.press/v108/park20b/park20b.pdf).

## Dataset

MNIST, USPS, SVHN, and Office31 datasets are required. To this end, run initializatio scripts in ```data/setup```. For example, to create MNIST dataset, run 
```
python3 init_MNIST_dataset.py
```

## Run Experiments

Experiment scripts for digit datasets and office31 dataset are located at ```demo/digits``` and ```demo/office31```, respectively. For example, FL+IW+Temp. experiments for the shift from MNIST to USPS, 
```
cd demo/digits/best_classifier/MNIST2USPS
ln -s ../../../../data/setup/ datasets
python3 exp_Temp_FL_IW.py
```
Note that the pretrained classifiers over source are included at ```snapshots``` and loaded during the experiments. 

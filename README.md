# DistBO
Python code for the following paper (partially adapted from https://github.com/fmfn/BayesianOptimization):

Ho Chung Leon Law, Peilin Zhao, Lucian Chan, Junzhou Huang, Dino Sejdinovic,
Hyperparameter Learning via Distributional Transfer, NeurIPS 2019.
https://arxiv.org/abs/1810.06305

## Setup
To setup as a package, clone the repository and run
```
python setup.py develop
```
This package also requires TensorFlow (tested on v1.7.0) to be installed.

## Structure
The directory is organised as follows:
* __distBO__: contains the main code
* __experiment__: contains the API code (train_test.py) and experimental configuration code to generate the cmd line
* __protein__: contains the pre-processed protein dataset 
* __parkinson__: contains the pre-processed parkinsons dataset (https://archive.ics.uci.edu/ml/datasets/parkinsons)

## Protein dataset
The protein-ligand binding data are extracted from ChEMBL 25. Seven different targets (ADAM17, AKT1, BRAF, COX1, FXA, GR, VEGFR2) are selected. It is labelled as active (1) if the binding affinity is below the thresholds, else is labelled as inactive (0). The thresholds vary across different protein families (see below). MACCS keys fingerprint are used as the feature and the implemention in RDKits is used.

Activity Thresholds:  
*  __Kinases__: <= 30nM    
*  __GPCRs__: <= 100nM   
* __Nuclear Receptors__: <= 100nM   
* __Ion Channels__: <= 10μM
 

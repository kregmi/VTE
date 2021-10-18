# 
# Setup

## Dependencies

1. [PyTorch](https://pytorch.org/get-started/locally/) (at least v0.4.0)
2. [scipy](https://www.scipy.org/)
    - [numpy](http://www.numpy.org/)
    - [sklearn](https://scikit-learn.org/stable/)
    - [h5py](https://www.h5py.org/)
3. [geopy](https://pypi.org/project/geopy)

## Data

Running this code requires a copy of BDD video frames (available [here](https://bdd-data.berkeley.edu/)), 
and the corresponding Google StreetView Images.
`data_loader_bdd_gsv_feats.py` contains the dataloader to load data for training/testing.


# Usage

`train.py` contains the code for training.
`eval.py` contains the code for testing.


## Train

Code to train a model on encoder features for BDD-GSV dataset.
Takes the output of geo-temporal feature learning network as input.

## Test

Code to test a previously trained model on encoder features for BDD-GSV dataset.
Takes the output of geo-temporal feature learning network as input.

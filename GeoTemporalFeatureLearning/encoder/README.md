# 
# Setup

## Dependencies

1. [PyTorch](https://pytorch.org/get-started/locally/) (at least v0.4.0)
2. [scipy](https://www.scipy.org/)
    - [numpy](http://www.numpy.org/)
    - [sklearn](https://scikit-learn.org/stable/)
    - [h5py](https://www.h5py.org/)
3. [tensorboardX](https://github.com/lanpa/tensorboardX)
4. [geopy](https://pypi.org/project/geopy)

## Data

Running this code requires a copy of BDD video frames (available [here](https://bdd-data.berkeley.edu/)), 
and the corresponding Google StreetView Images.
`bdd.py` contains the dataloader to load images for training/testing.


# Usage

`main.py` contains the code for training.


## Train

Code to train a model on BDD-GSV dataset.

## Test

Code to test a previously trained model on BDD-GSV dataset.
# DD2358 Final Project - Optimisation of Artificial Neural Network for Image Classification by pmocz 

## Team members
- Chong Wen Xuan Darryl
- Ching Xin Wei

## Overview
For our DD2358 Final Project, we seek to optimise the Artificial Neural Network for Image Classification algorithm by pmocz ([link](https://github.com/pmocz/artificialneuralnetwork-python)) using techniques we learnt throughout the course. 

More specifically, we utilised the following approaches:
- Loop Optimisation
- PyTorch
- Cython


## Running the code

### Dependencies
Make sure the following libraries are installed:
- PyTorch
- Cython
- matplotlib
- NumPy

#### To run the original code written by pmocz:
```
python artificialneuralnetwork.py
```

#### To run the Optimised Loop version:
```
python improveforloop.py
```

#### To run the PyTorch version:
```
python withpytorch.py
```

#### To run the Cython version:
1. First compile the Python code with Cython:
```
python setup.py build_ext --inplace
```
2. Then run the following:
```
python withcython.py
```


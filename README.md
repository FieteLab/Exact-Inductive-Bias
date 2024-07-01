# Exact-Inductive-Bias

This repository contains the code for the paper "Towards Exact Computation of Inductive Bias" in IJCAI 2024. In this paper, we propose a novel method for efficiently computing the inductive bias required for generalization on a task with a fixed training data budget. Our method provides a direct estimate of inductive bias without using bounds and is applicable to diverse hypothesis spaces. This repository contains code to evaluate inductive bias required to generalize on a dataset under both a kernel-based hypothesis space and a neural network hypothesis space.

## Requirements

The code is written in Python 3. To install the required packages, run:

```
pip install torch numpy scipy
```

## Files

- `kernel_inductive_bias.py`: Code to compute inductive bias for a kernel-based hypothesis space.'
- `nn_inductive_bias.py`: Code to compute inductive bias for a neural network hypothesis space.'
- `utils.py`: Utility functions for computing inductive bias.'

## Usage
Import the 'main' function from either `kernel_inductive_bias.py` or `nn_inductive_bias.py` and pass the required arguments to compute the inductive bias for a given dataset.

The required arguments for kernel-based hypothesis space are:

- `X_train`: Training data
- `y_train`: Training labels
- `X_test`: Test data
- `y_test`: Test labels
- `out_dim`: Output dimension of the task
- `target_error`: Target error rate

The required arguments for neural network hypothesis space are:
- `train_dataset`: Training dataset
- `test_dataset`: Test dataset
- `input_size`: Input size
- `output_size`: Output size
- `width`: Width of the neural network hidden layers
- `depth`: Number of hidden layers
- `epochs`: Number of training epochs
- `batch_size`: Batch size
- `num_trials`: Number of hypothesis samples
- `target_error`: Target error rate

## Example

```python
from kernel_inductive_bias import main as kernel_main
from nn_inductive_bias import main as nn_main

# Kernel-based hypothesis space
inductive_bias = kernel_main(X_train, y_train, X_test, y_test, out_dim, target_error)

# Neural network hypothesis space
inductive_bias = nn_main(train_dataset, test_dataset, input_size, output_size, width, depth, epochs, batch_size, num_trials, target_error)
```


## Citations
If you use this code for your research, please cite our paper.
```
@inproceedings{boopathy2024towards,
    author = {Boopathy, Akhilan and Yue, William and Hwang, Jaedong and Iyer, Abhiram and Fiete, Ila},
    title = {Towards Exact Computation of Induction Bias},
    booktitle = {IJCAI},
    year = {2021},
}   
```

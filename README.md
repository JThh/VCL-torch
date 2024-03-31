# Applying Variational Continual learning (VCL) to CNNs and ResNets: when it shines brighter than other methods

## Description

This is the official code repository for paper `Applying Variational Continual learning (VCL) to CNNs and ResNets: when it shines brighter than other methods`. 

It contains variational implementation of CNNs and ResNets, and other utility modules to facilitate other variational layer constructions (`./src/layers/` and `./src/models`). We provide notebooks and experimental scripts of variational networks on Permuted and Split CIFAR10 and CIFAR100 datasets at the `./src` folder.

## Experiments

Run `python ./src/main.py -h` that returns all possible experiment settings.

If you want to run all experiments, execute `python ./src/main.py all`. It should run in both GPU and CPU-only environments.

## Experimental Records.

We have shared our TensorBoard logs [here](https://drive.google.com/drive/folders/1R4JAHrVHfhd9qios3QmoBt0moWty5jLd?usp=sharing) of all finished experiments. The logs may contain some unfinished experiments. Kindly refer to the Appendix section of our paper for valid run names.

The table below resides all experimental settings we have run:

| Model Type | Method                       | Dataset                          | Task Type       |
|------------|------------------------------|----------------------------------|-----------------|
| CNN        | Discriminative VCL           | CIFAR10, CIFAR100, MNIST, notMNIST | Split, Permuted |
| CNN        | Elastic Weight Consolidation | CIFAR10, CIFAR100                | Split           |
| CNN        | Generative VCL               | MNIST                            | Split               |
| CNN        | Joint (Upperbound)    | CIFAR10, CIFAR100                | Split           |
| CNN        | Laplace Propagation          | CIFAR10, CIFAR100                | Split               |
| CNN        | Max Likelihood Estimation    | CIFAR100, MNIST                  | Split           |
| CNN        | Synaptic Intelligence        | CIFAR10, CIFAR100, notMNIST, MNIST | Split         |
| MLP        | Discriminative VCL           | CIFAR10, CIFAR100, MNIST, notMNIST | Split, Permuted |
| MLP        | Elastic Weight Consolidation | CIFAR10, CIFAR100, MNIST, notMNIST | Split           |
| MLP        | Max Likelihood Estimation    | CIFAR10, CIFAR100, MNIST         | Split           |
| MLP        | Synaptic Intelligence        | notMNIST                         | Split           |
| ResNet     | Discriminative VCL           | CIFAR10, CIFAR100                | Split           |
| ResNet     | Elastic Weight Consolidation | CIFAR10, CIFAR100                | Split           |
| ResNet     | Joint (Upperbound)     | CIFAR10, CIFAR100                | Split           |
| ResNet     | Laplace Propagation          | CIFAR10, CIFAR100                | Split               |

## Credits & Citation

The code was extended from [this repository](https://github.com/NixGD/variational-continual-learning).



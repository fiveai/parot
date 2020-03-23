# PaRoT: A Practical Framework for Robust Deep Neural Network Training

#### Authors: Edward W. Ayers, Francisco Eiras, Majd Hawasly, Iain Whiteside

## Installing PaRoT

We recommend installing PaRoT in a virtual environment (e.g. using Anaconda).
You can install it by running:
```
pip install .
```
from the main directory inside the repository. 

## Running Examples

The examples provided with the framework can be found inside the `examples`
folder of the package.

The simplest example corresponds to the training of a simple 5 layers neural
network on MNIST, which showcases the ease of use of our framework. It can be
launched by running:
```
cd examples
python3 train_simple_example.py
```
outputting a checkpoint which can then be loaded and tested. 

In `diffai_comparison.py` is the code corresponding to the paper experiments
which compare PaRoT to DiffAI. Each of the cases can be run by:
```
python3 diffai_comparison.py --model [MODEL_ID] --domain [DOMAIN_ID] --property [PROPERTY_ID] --dataset [DATASET_ID]
```
where `MODEL_ID` can be any of the models in the paper, `DOMAIN_ID` can be `box` or `hz` (Hybrid Zonotope in the paper) for the
built-in domains, `PROPERTY_ID` can be `ball`, `brightness` or `fourier` for example, and `DATASET_ID` is
either `MNIST` or `CIFAR10`. The comparsion results are outputted to a JSON file.

---

Copyright 2020 FiveAI Ltd. All rights reserved. PaRoT is released under the
"MIT License Agreement". Please see the LICENSE file that is included as part of
this package.

# What's this
Implementation of Residual Networks of Residual Networks by chainer  

# Dependencies

    git clone https://github.com/nutszebra/residual_networks_of_residual_networks.git
    cd residual_networks_of_residual_networks
    git submodule init
    git submodule update

# How to run
    python main.py -g 0

# Details about my implementation
All hyperparameters and network architecture are the same as in [[1]][Paper] except for data-augmentation.  
* Data augmentation  
Train: Pictures are randomly resized in the range of [32, 36], then 32x32 patches are extracted randomly and are normalized locally. Horizontal flipping is applied with 0.5 probability.  
Test: Pictures are resized to 32x32, then they are normalized locally. Single image test is used to calculate total accuracy.  

* Stochastic depth  
implemented

# Cifar10 result
| network                                   | total accuracy (%) |
|:------------------------------------------|-------------------:|
| RoR-3-WRN40-2+SD [[1]][Paper]             | 95.41              |
| my implementation                         | soon               |

<img src="https://github.com/nutszebra/residual_networks_of_residual_networks/blob/master/loss.jpg" alt="loss" title="loss">
<img src="https://github.com/nutszebra/residual_networks_of_residual_networks/blob/master/accuracy.jpg" alt="total accuracy" title="total accuracy">

# References  
Residual Networks of Residual Networks: Multilevel Residual Networks [[1]][Paper]

[paper]: https://arxiv.org/abs/1608.02908 "Paper"

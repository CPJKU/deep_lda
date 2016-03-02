Deep Linear Discriminant Analysis (DeepLDA)
===========================================

This repository contains code for reproducing the experiments reported in the ICLR 2016 paper
`Deep Linear Discriminant Analysis <http://arxiv.org/abs/1511.04707>`_
by Matthias Dorfer, Rainer Kelz and Gerhard Widmer

#### Requirements

The implementation is based on `Theano <https://github.com/Theano/Theano>`_
and the neural networks library `Lasagne <https://github.com/Lasagne/Lasagne>`_.
For installing Theano and Lasagne please follow the installation instruction on the respective github pages.

#### Experiments

We report results for three different benchmark datasets in our paper wich can be reproduced with the following commands:

##### Training

```
python exp_dlda.py --model mnist_dlda --data mnist --train
python exp_dlda.py --model cifar10_dlda --data cifar10 --train
python exp_dlda.py --model stl10_dlda --data stl10 --train --fold 0
```

##### Evaluation

```
python exp_dlda.py --model mnist_dlda --data mnist --eval
python exp_dlda.py --model cifar10_dlda --data cifar10 --eval
python exp_dlda.py --model stl10_dlda --data stl10 --eval --fold 0
```

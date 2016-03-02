Deep Linear Discriminant Analysis (DeepLDA)
===========================================

This repository contains code for reproducing the experiments reported in the ICLR 2016 paper
`Deep Linear Discriminant Analysis <http://arxiv.org/abs/1511.04707>`_
by Matthias Dorfer, Rainer Kelz and Gerhard Widmer from the `Department of Computational Perception <http://www.cp.jku.at/>`_ at JKU Linz.

Requirements
------------

The implementation is based on `Theano <https://github.com/Theano/Theano>`_
and the neural networks library `Lasagne <https://github.com/Lasagne/Lasagne>`_.
For installing Theano and Lasagne please follow the installation instruction on the respective github pages.

Experiments
-----------

We report results for three different benchmark datasets in our paper.
The datasets will be downloaded automatically form the corresponding data set pages.

Training
~~~~~~~~

To train the models just run the following commands:

MNIST: the model should train up to a validation accuracy of around 99.7%.::

    python exp_dlda.py --model mnist_dlda --data mnist --train

CIFAR10: the model should train up to 92% validation accuracy.::

    python exp_dlda.py --model cifar10_dlda --data cifar10 --train

Train on the first fold of STL10::

    python exp_dlda.py --model stl10_dlda --data stl10 --train --fold 0

* the model should train up to a validation accuracy of around 67%.
* we train this model on NVIDIA Tesla K40 (12GB memory) so it might be to large for less powerful cards (in this case you could try to reduce the batch size).

Evaluation
~~~~~~~~~~

For evaluating the trained models run the following commands:
The script will report:

* The accuracies on train, validation and test set
* Report the magnitudes of the individual eigenvalues after solving the general (Deep)LDA eigenvalue problem compare Figure 5 in paper)
* Produce some plots visualizing the structure of the latent representation produced by the model (compare Figure 5 in paper)
::

    python exp_dlda.py --model mnist_dlda --data mnist --eval
    python exp_dlda.py --model cifar10_dlda --data cifar10 --eval
    python exp_dlda.py --model stl10_dlda --data stl10 --eval --fold 0


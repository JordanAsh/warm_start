# On Warm-Starting Neural Network Training
This repository contains code to reproduce results from our 2020 NeurIPS paper [On Warm-Starting Neural Network Training](https://arxiv.org/abs/1910.08475). In it, we study the batch, online learning setting. Each time new data are added to the training set, one could either re-initialize model parameters from scratch (random initialization), initialize using parameters found by the previous round of optimization (warm start), or use our proposed strategy, which we call *shrink-perturb* initialization.


# Dependencies

To run this code you'll need [PyTorch](https://pytorch.org/) (we're using version 1.11.0).

# Running an experiment

`python run.py --model resnet --n_samples 1000 --lr 1e-2 --opt adam --shrink 0.4 --perturb 0.1`\
runs a batch online learning experiment using an 18-layer ResNet, Adam optimizer, and a learning rate of 1e-2. At each round, 1,000 samples are added to the training set and parameters are initialized using a shrinkage coefficient of 0.4 and a noise scale of 0.1.

Note that `--shrink 0 --perturb 1` is equivalent to a pure random initialization and `--shrink 1 --perturb 0` is equivalent to pure warm starting. See the paper for more details on these parameters.

`python run.py --model mlp --n_samples 0.5 --lr 1e-1 --lr_2 1e-4 --opt sgd --shrink 1 --perturb 0`\
runs a warm-start experiment using a multilayer-perceptron and SGD optimization. When `n_samples` is less than 1, we execute a two-phased learning experiment: in the first phase the model has access only to the specified percentage of training data (in this case 50%), and in the second phase the model has access to all training data.

`lr_2` specifies the learning rate for the second round of training. If omitted, the second-phase optimizer inherits the learning rate from `lr`. Analogous arguments are available for batch size and weight decay.

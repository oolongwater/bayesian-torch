#!/bin/bash

# Model parameters
model=bayesian_rnn  # Name of the RNN-to-BNN model
mode='train'        # Mode: 'train' or 'test'
batch_size=64       # Batch size for training/testing
num_epochs=20       # Number of epochs for training
num_monte_carlo=50  # Number of Monte Carlo samples for uncertainty estimation
learning_rate=0.001 # Learning rate for training

# Training the Bayesian RNN model
if [ "$mode" == "train" ]; then
    python examples/main_bayesian_rnn.py \
        --arch=$model \
        --mode=$mode \
        --batch-size=$batch_size \
        --epochs=$num_epochs \
        --lr=$learning_rate
fi

# Testing the Bayesian RNN model
if [ "$mode" == "test" ]; then
    python examples/main_bayesian_rnn.py \
        --arch=$model \
        --mode=$mode \
        --batch-size=$batch_size \
        --num_monte_carlo=$num_monte_carlo
fi
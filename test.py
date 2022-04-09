#!/usr/bin/env python3.9

import random

from platform import node
from nnlib import *

network_shape: list[int] = [2, 2, 1]
activation: DerivableFunction = Activations.Sigmoid
output_activation: DerivableFunction = Activations.Sigmoid
regularization: DerivableFunction = None
input_ids: list[str] = ['bit_a','bit_b']
init_zero: bool = False

batch: list = [
    ([0.0,0.0],[0.0]),
    ([0.0,1.0],[1.0]),
    ([1.0,0.0],[1.0]),
    ([1.0,1.0],[0.0]),
]

inputs: list[float] = [0, 0]
target: list[float] = [0]

error: DerivableFunction = Errors.Square

learning_rate: float = 0.03
regularization_rate: float = 0.0

neural: Network = Network(network_shape, activation,
    output_activation, regularization, input_ids, init_zero)

for i in range(1000000):
    sample: object=random.choice(batch)
    neural.forwardPropagation(sample[0])

    #print(f"{sample[0]}: {sample[1]} -> {neural.outputs}")

    neural.backwardPropagation(sample[1],error)
    neural.updateWeights(learning_rate, regularization_rate)

print()

for sample in batch:
    neural.forwardPropagation(sample[0])
    print(f"{sample[0]}: {sample[1]} -> {neural.outputs}")

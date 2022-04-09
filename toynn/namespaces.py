
import math

from collections import namedtuple

__all__ = [
    'Errors',
    'Activations',
    'Regularizations'
]

class Namespace(object):
    pass

class DerivableFunction(namedtuple("DerivableFunction","function derivative")):
    pass

Errors: Namespace=Namespace()

Errors.Square=DerivableFunction(
    (lambda output, target: 0.5 * math.pow(output - target, 2)),
    (lambda output, target: output - target),
)

Activations: Namespace=Namespace()

Activations.Tanh=DerivableFunction(
    (lambda input: math.tanh(input)),
    (lambda input: 1 - (math.tanh(input)**2)),
)

Activations.Relu=DerivableFunction(
    (lambda input: max(0, input)),
    (lambda input: 0 if input <= 0 else 1),
)

Activations.Sigmoid=DerivableFunction(
    (lambda input: 1 / (1 + math.exp(-input))),
    (lambda input: (1 / (1 + math.exp(-input))) * (1 - (1 / (1 + math.exp(-input))))),
)

Activations.Linear=DerivableFunction(
    (lambda input: input),
    (lambda input: 1),
)

Regularizations: Namespace=Namespace()

Regularizations.L1=DerivableFunction(
    (lambda weight: abs(weight)),
    (lambda weight: -1 if weight < 0 else 1 if weight > 1 else 0),
)

Regularizations.L2=DerivableFunction(
    (lambda weight: 0.5 * (weight**2)),
    (lambda weight: weight),
)


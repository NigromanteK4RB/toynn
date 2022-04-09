
import random

from .namespaces import Activations
from .namespaces import DerivableFunction
from .namespaces import Errors
from .namespaces import Regularizations

__all__: list[str] = [
    'Activations',
    'DerivableFunction',
    'Errors',
    'Regularizations',
    'Network',
]

class Node(object):
    id: str

    input_links: list
    output_links: list

    bias: float
    total_input: float
    output: float
    output_derivative: float
    input_derivative: float
    acc_input_derivative: float
    num_accumulated_derivatives: int
    activation: DerivableFunction

    def __init__(node: object, id: str, activation: DerivableFunction, init_zero: bool = False) -> None:
        node.id = id
        node.activation = activation
        node.bias: float=random.random() * 2 - 1

        node.input_links: list[Link]=list()
        node.output_links: list[Link]=list()

        node.total_input: float=float()
        node.output: float=float()
        node.output_derivative: float=float()
        node.input_derivative: float=float()
        node.acc_input_derivative: float=float()
        node.num_accumulated_derivatives: int=int()

        if init_zero:
            node.bias = float()

    def updateOutput(node) -> float:
        node.total_input = node.bias

        for link in node.input_links:
            node.total_input += link.weight * link.source.output

        node.output = node.activation.function(node.total_input)

        return node.output

class Link(object):
    id: str
    source: Node
    dest: Node
    weight: float
    is_dead: bool
    error_derivative: float
    accumulated_error_derivative: float
    num_accumulated_derivatives: int
    regularization: DerivableFunction

    def __init__(link: object, source: Node, dest: Node,
                 regularization: DerivableFunction, init_zero: bool = False) -> None:
        link.id: str = source.id + "-" + dest.id
        link.source: Node = source
        link.dest: Node = dest
        link.regularization: DerivableFunction = regularization
        link.weight: float = random.random() * 2 -1
        link.is_dead: bool = False
        link.error_derivative: float = float()
        link.accumulated_error_derivative: float = float()
        link.num_accumulated_derivatives: int = int()

        if init_zero:
            link.weight = float()



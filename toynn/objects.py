
import random

from .namespaces import Activations
from .namespaces import DerivableFunction
from .namespaces import Errors
from .namespaces import Regularizations

from typing import Callable

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

class Layer(list[Node]):
    @property
    def output(layer: list[Node]) -> list[float]:
        return [node.output for node in layer]

class Network(list[Layer]):

    @property
    def output_layer(network: list[Layer]) -> Layer:
        return network[-1]

    @property
    def input_layer(network: list[Layer]) -> Layer:
        return network[0]

    def forEachNode(network: list[Layer], ignore_inputs: bool, accessor: Callable) -> None:
        for layer in network[1 if ignore_inputs else 0:]:
            for node in layer:
                accessor(node)

    @property
    def outputs(network: list[Layer]) -> list[float]:
        return network.output_layer.output

    def __init__(network: list[Layer], shape: list[int], hidden_activation: DerivableFunction,
                 output_activation: DerivableFunction, regularization: DerivableFunction,
                 input_ids: list[str], init_zero: bool = False) -> None:
        layers: int=len(shape)
        id: int=1

        for layer in range(layers):
            is_output: bool= layer == (layers - 1)
            is_input: bool=layer == 0

            currentl: Layer=Layer()

            network.append(currentl)

            nodes: int=shape[layer]

            for node in range(nodes):
                currentn_id: int=str(id)
                if (is_input):
                    currentn_id = input_ids[node]
                else:
                    id += 1

                currentn: Node=Node(currentn_id, output_activation if is_output else hidden_activation, init_zero)

                currentl.append(currentn)

                if (layer >= 1):
                    for prevn in network[layer - 1]:
                        new: Link=Link(prevn, currentn, regularization, init_zero)
                        prevn.output_links.append(new)
                        currentn.input_links.append(new)

    def forwardPropagation(network: list[Layer], inputs: list[float]) -> list[float]:
        if (len(inputs) != len(network.input_layer)):
            raise ValueError("The number of inputs must match the number of nodes in the input layer")

        for n, node in enumerate(network.input_layer):
            node.output = inputs[n]

        for layer in network[1:]:
            for node in layer:
                node.updateOutput()

        return network.outputs

    def backwardPropagation(network: list[Layer], target: list[float], error: DerivableFunction) -> None:
        for n, node in enumerate(network.output_layer):
            node.output_derivative = error.derivative(node.output, target[n])

        for ln, currentl in enumerate(r_network := list(reversed(network[1:]))):
            for node in currentl:
                node.input_derivative = node.output_derivative * node.activation.derivative(node.total_input)
                node.acc_input_derivative += node.input_derivative
                node.num_accumulated_derivatives +=1

            for node in currentl:
                for link in node.input_links:
                    if (link.is_dead):
                        continue
                    link.error_derivative = node.input_derivative * link.source.output
                    link.accumulated_error_derivative += link.error_derivative
                    link.num_accumulated_derivatives += 1

            if (currentl == r_network[-1]):
                continue

            prevl: Layer=r_network[ln+1]

            for node in prevl:
                node.output_derivative = 0
                for link in node.output_links:
                    node.output_derivative += link.weight * link.dest.input_derivative

    def updateWeights(network: list[Layer], learning_rate: float, regularization_rate: float) -> None:
        for layer in network:
            for node in layer:
                if node.num_accumulated_derivatives > 0:
                    node.bias -= learning_rate * (node.acc_input_derivative / node.num_accumulated_derivatives)
                    node.acc_input_derivative = 0
                    node.num_accumulated_derivatives = 0

                for link in node.input_links:
                    if link.is_dead:
                        continue

                    if link.regularization != None:
                        regularization_derivative = link.regularization.derivative(link.weight)
                    else:
                        regularization_derivative = 0

                    if link.num_accumulated_derivatives > 0:
                        link.weight -= (learning_rate / link.num_accumulated_derivatives) * link.accumulated_error_derivative

                    new_link_weight = link.weight - (learning_rate * regularization_rate * regularization_derivative)

                    if ((link.regularization == Regularizations.L1) and (link.weight * new_link_weight) < 0):
                        link.weight = 0
                        link.is_dead = True
                    else:
                        link.weight = new_link_weight

                    link.accumulated_error_derivative = 0
                    link.num_accumulated_derivatives = 0


import random
from autograd.engine import Scalar

class Module(object):
    '''Base class'''

    def zero_grad(self):
        '''Zero out gradients before backpropagation in order to clear out 
        accumulated gradients from previous error/loss.'''
        for param in self.parameters():
            param.grad = 0

    def parameters(self):
        '''Override this method.'''
        return []

class Neuron(Module):
    '''Represents one computational unit in the network.'''

    def __init__(self, num_inputs, non_linear=True):
        # random initialization of weights
        self.weights = [Scalar(random.uniform(-1,1)) for _ in range(num_inputs)]
        self.bias = Scalar(0)
        self.non_linear = non_linear

    def __call__(self, x):
        '''Compute the neuron's activation by calculating the dot product of the weights and inputs, 
        adding the bias, and optionally passing through a non-linear activation function like ReLU.'''
        dot_product = sum((w_i * x_i for w_i, x_i in zip(self.weights, x)))
        output = dot_product + self.bias
        activation = output.relu() if self.non_linear else output
        return activation

    def __repr__(self):
        return f"{'ReLU' if self.non_linear else 'Linear'}Neuron({len(self.weights)})"

    def parameters(self):
        return self.weights + [self.bias]

class Layer(Module):
    '''Represents one layer of neurons in the network.'''

    def __init__(self, num_inputs, num_outputs, **kwargs):
        self.neurons = [Neuron(num_inputs, **kwargs) for _ in range(num_outputs)]
    
    def __call__(self, x):
        output = [neuron(x) for neuron in self.neurons]
        return output[0] if len(output) == 1 else output

    def __repr__(self):
        return f"Layer([{','.join(str(neuron) for neuron in self.neurons)}])"

    def parameters(self):
        return [parameter for neuron in self.neurons for parameter in neuron.parameters()]
    
class MLP(Module):
    '''
    Feed-forward multi-layer perceptron.
    
    Example:
    
    mlp = MLP(2, [16,16,1])

    - 2 dimensional input X
    - 2 layers with 16 hidden units each
    - 1 dimensional output (binary classification)
    '''

    def __init__(self, num_inputs, hidden_units):
        sizes = [num_inputs] + hidden_units
        self.layers = [Layer(sizes[i], sizes[i+1], non_linear=(i!=len(hidden_units)-1)) for i in range(len(hidden_units))]

    def __call__(self, x):
        # forward propagation
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        return f"MLP([{','.join(str(layer) for layer in self.layers)}])"

    def parameters(self):
        return [parameters for layer in self.layers for parameters in layer.parameters()]    

'''
A scalar valued autograd engine which implements backpropagation over 
a dynamically built directed acyclical graph (DAG).
'''

import math

class Scalar(object):
    '''Represents a scalar value and its gradients.'''

    def __init__(self, value, parent_nodes=[], prev_op=None):
        assert isinstance(value, (float, int)), "node value must be a scalar"
        self.value = value
        self.parent_nodes = parent_nodes
        self.prev_op = prev_op  # operator which created this node (i.e. add, sub, mul, etc.)
        self.grad = 0           # stores derivative of output with respect to self
        self.grad_wrt = {}      # stores derivatives of self with respect to each parent (wrt = "with respect to")

    def __repr__(self):
        return f'Scalar(value={self.value:.2f}, grad={self.grad:.2f}), prev_op={self.prev_op}'

    # Called by: self + other_node
    def __add__(self, other_node):
        if not isinstance(other_node, Scalar):
            other_node = Scalar(other_node)

        # perform arithmetic operation and generate output node
        output_node = Scalar(self.value + other_node.value, [self, other_node], '+')

        # derivative of output with respect to self is 1
        output_node.grad_wrt[self] = 1

        # derivative of output with respect to other node is 1
        output_node.grad_wrt[other_node] = 1

        return output_node

    # Called by: other_node + self
    def __radd__(self, other_node):
        # NOTE: we can use __add__ here because the derivatives of the
        # output node with respect to self and other_node will both be 1
        # (i.e. the communative property holds for addition)
        return self.__add__(other_node)

    # Called by: self - other_node
    def __sub__(self, other_node):
        if not isinstance(other_node, Scalar):
            other_node = Scalar(other_node)

        # perform arithmetic operation and generate output node
        output_node = Scalar(self.value - other_node.value, [self, other_node], '-')

        # derivative of output with respect to self is 1 
        # (i.e. derivative of x is 1)
        output_node.grad_wrt[self] = 1

        # derivative of output with respect to other node is -1 
        # (i.e. derivative of -x is -1)
        output_node.grad_wrt[other_node] = -1

        return output_node

    # Called by: other_node - self
    def __rsub__(self, other_node):
        # NOTE: we CANNOT use __sub__ here because the element
        # being subtracted will have a derivative of -1, so order matters.
        # (i.e. the communative property DOES NOT hold for subtraction)
        if not isinstance(other_node, Scalar):
            other_node = Scalar(other_node)

        # perform arithmetic operation and generate output node
        output_node = Scalar(other_node.value - self.value, [self, other_node], '-')

        # derivative of output with respect to self is =1 
        # (i.e. deravative of -x is -1)
        output_node.grad_wrt[self] = -1

        # derivative of output with respect to other node is 1 
        # (i.e. deravative of x is 1)
        output_node.grad_wrt[other_node] = 1

        return output_node

    # Caled by: self * other_node
    def __mul__(self, other_node):
        if not isinstance(other_node, Scalar):
            other_node = Scalar(other_node)

        # perform arithmetic operation and generate output node
        output_node = Scalar(self.value * other_node.value, [self, other_node], '*')

        # derivative of output with respect to self will be other_node.value 
        # (i.e. z=x*y, dz/dx = y)
        output_node.grad_wrt[self] = other_node.value

        # derivative of output with respect to self will be other_node.value 
        # (i.e. z=x*y, dz/dy = x)
        output_node.grad_wrt[other_node] = self.value

        return output_node

    # Called by: other_node * self
    def __rmul__(self, other_node):
        # NOTE: we can use __mul__ here because the derivatives of the
        # output node with respect to self and other_node are not affected by order
        # (i.e. the communative property holds for multiplication)
        return self.__mul__(other_node)

    # Called by: self / other_node
    def __truediv__(self, other_node):
        if not isinstance(other_node, Scalar):
            other_node = Scalar(other_node)
        
        # perform arithmetic operation and generate output node
        output_node = Scalar(self.value / other_node.value, [self, other_node], '/')

        # derivative of output with respect to self will be 1/other_node.value
        # (i.e. z=x/y, dz/dx=1/y)
        output_node.grad_wrt[self] = 1/other_node.value
        
        # derivative of output with respect to self is calculated using the Power Rule
        #  z=x/y -> x*y^-1 -> dz/dy = -x*y^-2 -> -x/y^2
        output_node.grad_wrt[other_node] = -self.value / other_node.value**2

        return output_node

    # Called by: other_node / self
    def __rtruediv__(self, other_node):
        if not isinstance(other_node, Scalar):
            other_node = Scalar(other_node)
        
        # perform arithmetic operation and generate output node
        output_node = Scalar(other_node.value / self.value, [self, other_node], '/')

        # derivative of output with respect to self is calculated using the Power Rule
        # (i.e. z=y/x -> y*x^-1 -> dz/dx = -y*x^-2 -> -y/x^2)
        output_node.grad_wrt[self] = -other_node.value / self.value**2

        # derivative of output with respect to self will be 1/self.value
        # (i.e. z=y/x, dz/dy=1/x)
        output_node.grad_wrt[other_node] = 1/self.value
        
        return output_node

    # Called by: self ** x (i.e. self^x in math notation)
    def __pow__(self, x):
        assert isinstance(x, (int, float)), "power must be int or float"
        
        # perform exponentiation and create output node
        output_node = Scalar(self.value ** x, [self], f'^{x}')

        # derivative of output with respect to self is calculated with the Power Rule
        output_node.grad_wrt[self] = x * self.value ** (x-1)

        # NOTE: x is a scalar value, not another node, so we don't store a gradient for it
        return output_node

    # Called by: -self
    def __neg__(self):
        return self.__mul__(-1)

    def relu(self):
        '''
        ReLU (Rectified Linear Unit): activation function for the node        
        A piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero.
        '''
        # perform relu activation function and create output node
        output_node = Scalar(max(0, self.value), [self], 'relu')

        # derivative of relu function will be 0 or 1
        output_node.grad_wrt[self] = int(self.value > 0)

        return output_node

    def sigmoid(self):
        '''
        Sigmoid activation function
        
        g(x) = 1 / (1 + e^-x)
        '''
        # perform sigmoid activation function and create output node
        g_x = 1 / (1 + math.e ** -self.value)
        output_node = Scalar(g_x, [self], 'sigmoid')

        # derivative of sigmoid can be defined as follows
        # g(x)  = 1 / (1 + e^-x)
        # g'(x) = g(x) * (1 - g(x))
        output_node.grad_wrt[self] = g_x * (1 - g_x)

        return output_node

    def backward(self):
        '''
        Compute the derivative of output with respect to all prior nodes using reverse-mode auto-differentiation.
        In order to do this, we traverse the DAG (Directed Acylical Graph) in a reversed topological order
        (i.e. from output to inputs) computing the gradient at each step.
        '''

        def _get_topological_order():
            '''Returns a linear ordering of self's vertices such that for every directed edge uv from vertex u to vertex v, u comes before v in the ordering.'''
            def _topological_sort(node):
                if node not in visited:
                    visited.add(node)
                    for parent in node.parent_nodes:
                        _topological_sort(parent)
                    ordered.append(node)

            ordered, visited = [], set()
            _topological_sort(self)
            return ordered

        def _compute_grad_of_parents(node):
            '''
            Compute the derivative of output with respect to each parent node.
            We can calculate this using the Chain Rule. 
            (i.e. doutput/dparent = doutput/dnode * dnode/dparent)
            '''
            for parent in node.parent_nodes:
                doutput_dnode = node.grad
                dnode_dparent = node.grad_wrt[parent]
                parent.grad += doutput_dnode * dnode_dparent

        self.grad = 1

        # traverse in reverse topological order
        for node in reversed(_get_topological_order()):
            _compute_grad_of_parents(node)
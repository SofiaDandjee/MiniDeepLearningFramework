from .base import Module
from ..initializers import Default
from torch import empty
import math


class TrainableLayer(Module):

    def forward(self, x):
        super().forward(x)
        
    def requires_grad(self):
        return True    


class Linear(TrainableLayer):
    
    def __init__(self, in_size, out_size, init = Default()):
        self.weights = empty(in_size, out_size)
        init.init_param(self.weights)

        self.bias = empty(1, out_size).normal_()
        self.x = 0
        self.dx1 = 0
        self.dw = 0
        self.db = 0
        
    def forward(self, x):
        super().forward(x)
        return (x @ self.weights) + self.bias
    
    def backward(self, *output_grad):
        """first item of output_grad should be the derivative of the previous layer"""
        self.dx1 = output_grad[0]
        self.dw = self.x.t() @ self.dx1
        self.db = self.dx1.sum(dim=0)
        return output_grad[0] @ self.weights.t()
    
    def step(self, eta):
        """
        Perform a SGD step

        Args:
            eta (float): weight of the gradient in the update
        """
        dw = self.x.t() @ self.dx1
        db = self.dx1
        self.weights.add_(-eta * self.dw)
        self.bias.add_(-eta * self.db)
        return
    
    def zero_grad(self):
        super().zero_grad()
        self.dw = 0
        self.db = 0
        
    def param(self):
        return [(self.weights, self.dw), (self.bias, self.db)]


class Sequential(TrainableLayer):
    
    def __init__(self, *layers):
        self.layers = layers
    
    def forward(self, x):
        super().forward(x)
        y = x
        for layer in self.layers:
            y = layer.forward(y)        
        return y
    
    def backward(self, *output_grad):
        d_list = [output_grad[0]]
        
        # We iterate from last layer to first layer
        # Each layer's gradient is then prepended to the gradient list
        for layer in self.layers[::-1]:
            d_list.insert(0, layer.backward(*d_list))            
        return d_list
    
    def step(self, eta):
        """
        Perform a SGD step

        Args:
            eta (float): weight of the gradient in the update
        """
        for layer in self.layers:
            if layer.requires_grad():
                layer.step(eta)
    
    def zero_grad(self):
        super().zero_grad()
        for layer in self.layers:
            layer.zero_grad()
            
    def param(self):
        return [layer.param() for layer in self.layers]
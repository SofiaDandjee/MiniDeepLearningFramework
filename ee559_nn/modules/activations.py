from .base import Module

class ActivationFunc(Module):

    def forward(self, x):
        super().forward(x)
    
    def requires_grad(self):
        return False
    
    def param(self):
        return []


class ReLU(ActivationFunc): 

    def forward(self, x):
        super().forward(x)
        return (x > 0) * x
    
    def backward(self, *output_grad):
        """first item of output_grad should be the derivative of the previous layer"""
        dl_dr = (self.x > 0) * 1.0 * output_grad[0]
        return dl_dr


class Sigmoid(ActivationFunc):
    
    def sigmoid(self, x):
        return 1 / (1 + (-x).exp())
    
    def forward(self, x):
        super().forward(x)
        return self.sigmoid(x)
    
    def backward(self, *output_grad):
        """first item of output_grad should be the derivative of the previous layer"""
        dl_ds = (self.sigmoid(self.x) * (1 - self.sigmoid(self.x))) * output_grad[0]
        return dl_ds


class LeakyReLU(ActivationFunc):       

    def __init__(self, alpha=0.01):
        self.alpha = alpha 
    
    def lrelu(self, x):
        return (x > 0) * x + (x <= 0) * self.alpha * x
    
    def forward(self, x):
        super().forward(x)
        return self.lrelu(x)
    
    def backward(self, *output_grad):
        """first item of output_grad should be the derivative of the previous layer"""
        inter = (self.x >=0) + (self.x < 0) * self.alpha
        dl_dlr =  inter * output_grad[0]
        return dl_dlr


class Tanh(ActivationFunc):
    
    def forward(self, x):
        super().forward(x)
        return (2 / (1 + (-2*x).exp())) - 1
    
    def backward(self, *output_grad):
        """first item of output_grad should be the derivative of the previous layer"""
        dl_dt = (1 - self.forward(self.x)**2) * output_grad[0]
        return dl_dt


class ELU(ActivationFunc):    

    def __init__(self, alpha=0.01):
        self.alpha = alpha    
    
    def forward(self, x):
        super().forward(x)
        return (self.x >0) * x + (self.x <= 0) * self.alpha * (self.x.exp() -1)
    
    def backward(self, *output_grad):
        """first item of output_grad should be the derivative of the previous layer"""
        inter = (self.x > 0) * 1.0 + (self.x <= 0) * self.alpha * self.x.exp()
        dl_del =  inter * output_grad[0]
        return dl_del


class SoftMax(ActivationFunc):
    
    def softmax(self, x):
        e = (x - x.max()).exp()
        return e.exp() / e.exp().sum(1).view(-1,1)
    
    def forward(self, x):
        super().forward(x)
        return self.softmax(x)
    
    def backward(self, *output_grad):
        """first item of output_grad should be the derivative of the previous layer"""
        
        s = self.softmax(self.x)
        dl_dsm = s * (1 - s)
        return dl_dsm * output_grad[0]
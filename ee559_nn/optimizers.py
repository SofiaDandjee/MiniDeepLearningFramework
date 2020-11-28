class Optimizer(object):
    """
    Base class of an optimizer
    """
    
    def __init__(self, eta, model):
        self.eta = eta
        self.model = model
        
    def step(self):
        pass

class SGD(Optimizer):
    """
    SGD optimizer
    /!\ only works on simple Sequential models
    """
    
    def __init__(self, eta, model, weight_decay=0):
        super().__init__(eta, model)
        self.weight_decay = weight_decay
        
    def step (self):
        for layer in self.model.layers:
            if layer.requires_grad():
                for param, grad in layer.param():
                    param.add_(-self.eta * grad - self.weight_decay * param)
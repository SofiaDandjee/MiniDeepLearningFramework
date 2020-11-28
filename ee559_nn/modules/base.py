class Module(object):
    def forward(self, x):
        self.x = x
    
    def backward(self, *output_grad):
        pass
    
    def requires_grad(self):
        pass
    
    def zero_grad(self):
        self.x= 0
    
    def param(self):
        pass
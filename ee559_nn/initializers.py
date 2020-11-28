import math

class Initializer(object):
    def init_param(self, param):
        pass

class Default(Initializer):
    def init_param(self, param):
        param.normal_()

class He(Initializer):
    def init_param(self, param):
        in_size, _ = param.shape
        std = math.sqrt(2.0 / in_size)
        param.normal_(std=std)

class Xavier(Initializer):
    def init_param(self, param):
        in_size, out_size = param.shape
        std = math.sqrt(1.0 / (in_size + out_size))
        param.normal_(std=std)
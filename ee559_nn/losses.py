class Loss(object):
    def loss(self, predicted, target):
        pass
    def dloss(self, predicted, target):
        pass


class MSELoss(Loss):
    def loss(self,predicted, target):
        return ((predicted - target)**2).mean()
    
    def dloss(self,predicted, target):
        return 2 * (predicted - target) / len(predicted)
    

class MAELoss(Loss):
    def loss(self,predicted, target):
        return (predicted - target).abs().mean()
    
    def dloss(self,predicted, target):
        e = (predicted - target)
        return -1.0 * (e < 0) + 1.0 * (e >= 0)


class CrossEntropyLoss(Loss):
    def softmax(x):
        a, _ = x.max(1)
        expo = (x - a.view(-1,1)).exp()
        return expo / expo.sum(dim=1).view(-1,1)
    
    def loss(self, predicted, target):
        a,_ = predicted.sum(dim=1)
        loss = - (predicted*target).sum(dim=1)
        loss += a
        loss += predicted - a.view(-1,1).log().exp().sum(dim=1)
        return loss.mean(dim=0)
    
    def dloss(self, predicted, target):
        dloss = self.softmax(predicted) - target
        return dloss / len(predicted)
import math
from torch import empty, rand
from .losses import MSELoss
from .optimizers import SGD

def compute_error_ratio(predicted, classes):
    _, predicted_classes = predicted.max(1)
    _, labels = classes.max(1)
    e = (predicted_classes - labels).float().abs().mean()
    return e


def generate_data(nb_samples):
    train_input = rand(nb_samples,2)
    train_target = empty(nb_samples).long()
    train_target = train_input.sub(0.5).pow(2).sum(1).sub(1 / (2*math.pi)).sign().add(1).div(2).float().view(-1,1)
    return train_input, train_target


def convert_to_one_hot_labels(target):
    tmp = empty(target.size(0), int(target.max().item()) + 1).zero_()
    tmp.scatter_(1, target.view(-1, 1).long(), 1.0)
    return tmp


def train_model(model, train_input, train_target, mini_batch_size=100, eta=0.01, iterations=1000, criterion=MSELoss(), optimizer=None):  
    if optimizer is None:
        optimizer = SGD(eta = eta, model = model, weight_decay = 0)
        
    training_loss = empty((iterations,1)) #Training loss over epochs
    training_error = empty((iterations,1)) #Training error over epochs
    for i in range(iterations):
        for inputs, targets in zip(train_input.split(mini_batch_size),
                                  train_target.split(mini_batch_size)):
        
            predicted = model.forward(inputs)
            loss = criterion.loss(predicted, targets)
            e = compute_error_ratio(predicted, targets)
        
            dloss = criterion.dloss(predicted, targets)
        
            model.backward(dloss)
            optimizer.step()
            model.zero_grad()
            
            training_loss[i] = loss
            training_error[i] = e
        
        if i%(iterations//10) == 0:
            print("Training Loss at {:04d}: {:3.3f} with {:02.2f}% of error        ".format(i, loss, e*100), end = '\r')

    return training_loss, training_error


def evaluate_model(model_gen,\
                   rounds=10,\
                   iterations = 1000,\
                   lr = 0.01,\
                   criterion=MSELoss(),\
                   num_samples=1000,\
                   mini_batch_size=100):

    test_error = empty((rounds,1))
    train_error = empty((rounds,1))
    training_loss = empty((rounds,iterations))
    for k in range(rounds):
        model = model_gen()
        
        #Generate data
        train_input, train_target = generate_data(num_samples)
        test_input, test_target = generate_data(num_samples)
        
        train_target = convert_to_one_hot_labels(train_target)
        test_target = convert_to_one_hot_labels(test_target)
        
        loss, _ = train_model(model, train_input, train_target, iterations = iterations, criterion=MSELoss())
        predicted_test = model.forward(test_input)
        predicted_train = model.forward(train_input)
        
        
        err_test = compute_error_ratio(predicted_test, test_target)
        err_train = compute_error_ratio(predicted_train, train_target)
        
        training_loss[k] = loss.t()
        train_error[k] = err_train
        test_error[k] = err_test
        
    return test_error.mean().item(), test_error.std().item(), train_error.mean().item(), train_error.std().item(), training_loss.mean(dim=0)
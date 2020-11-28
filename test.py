#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ee559_nn.modules.activations import *
from ee559_nn.modules.trainable_layers import *
from ee559_nn.losses import *
from ee559_nn.optimizers import *
from ee559_nn.utils import *
from ee559_nn.initializers import *


# In[2]:


train_input, train_target = generate_data(1000)
test_input, test_target = generate_data(1000)

train_target = convert_to_one_hot_labels(train_target)
test_target = convert_to_one_hot_labels(test_target)


# In[3]:


def model_gen(): 
    return Sequential(
        Linear(2, 25, init=Default())     , ReLU(),\
        Linear(25,25, init=Default()) , LeakyReLU(0.05),\
        Linear(25,25, init=Default())     , Tanh(),\
        Linear(25,25, init=Default()) , Sigmoid(),\
        Linear(25, 2, init=Default()), ELU(0.5)\
    )


# In[5]:


iterations = 2001
eta = 0.005
mini_batch_size = 500

model = model_gen()
criterion = MSELoss()
optimizer = SGD(eta=eta, model=model, weight_decay=0)

loss, train_error = train_model(model, train_input, train_target, mini_batch_size, eta, iterations, criterion, optimizer)
predicted_test = model.forward(test_input)
test_error = compute_error_ratio(predicted_test, test_target)
print()
print()
print("Train error: {:02.2f}%".format(train_error[-1].item()*100))
print("Test error : {:02.2f}%".format(test_error*100))


# In[ ]:





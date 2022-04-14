#!/usr/bin/env python
# coding: utf-8

# In[12]:


import tensorflow as tf
import numpy as np
from tensorflow import keras
#Simplest possible Neural Network. It has 1 layer, that layer has one neuron, and the input shape to its only one value
model = tf.keras.Sequential([keras.layers.Dense(units = 1,input_shape=[1])])
#We need to specify two functions: optimizer and loss function
#Loss function: Measures the guessed answers against the known correct answers and measures how well or badly it did
#optimizer function: Makes another guess. Based on the loss function it tries to minimize the loss
#The model repeats that for the number of epochs
#sgd: stochastic gradient descent
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')
#A Python library called NumPy provides lots of array type data structures to do this.
#Specify the values as an array in NumPy with np.array[]
xs = np.array([-1.0,0.0,1.0,2.0,3.0,4.0], dtype = float)
ys = np.array([-2.0, 1.0,4.0,7.0,10.0,13.0], dtype=float)
#Train the neural network: Learn the relationship between X's and Y's
#The process of training the neural network is in the model.fit call.
#That's where it go through a loop before making a guess, measuring how good or bad it is (the loss), 
#or using the optimizer to make another guess
# It will do that for the number of epochs you specify
# When you run that code, you'll see the loss will be printed out for each epoch
model.fit(xs, ys, epochs=500)


# In[13]:


#You have a model that has been trained to learn the relationship between X and Y.
#You can use the model.predict method to figure out the Y for a previously unknown X.
#For example, if X is 10, what do you think Y will be?
print(model.predict([10.0]))


# In[14]:


print(model.predict([5.0]))


# In[ ]:





# In[ ]:





# In[ ]:





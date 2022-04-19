#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


#You have a model that has been trained to learn the relationship between X and Y.
#You can use the model.predict method to figure out the Y for a previously unknown X.
#For example, if X is 10, what do you think Y will be?
print(model.predict([10.0]))


# In[3]:


print(model.predict([5.0]))


# In[4]:


# Save the entire model as a SavedModel.
get_ipython().system('mkdir -p saved_polynomial_TestingV2')
model.save('saved_polynomial_TestingV2/my_model')


# In[13]:


ls saved_polynomial_TestingV2


# In[15]:


ls saved_polynomial_TestingV2/my_model


# In[16]:


new_model = tf.keras.models.load_model('saved_polynomial_TestingV2/my_model')

# Check its architecture
new_model.summary()


# In[17]:


new_model.predict([10.0])


# In[23]:


# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('saved_polynomial_TestingV2/my_model') # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('saved_polynomial_TestingV2tflite.tflite', 'wb') as f:
    f.write(tflite_model)


# In[26]:


import tensorflow as tf
from tensorflow import keras
model = keras.models.load_model('saved_polynomial_TestingV2/my_model')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('saved_polynomial_TestingV2tflite_keras.tflite', 'wb') as f:
    f.write(tflite_model)


# In[55]:


import numpy as np
import tensorflow as tf

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="saved_polynomial_TestingV2tflite.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print("input data: " , input_data,"output data:" , output_data)


# In[53]:


#Loading the inference model

import numpy as np
import tensorflow as tf

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="saved_polynomial_TestingV2tflite_keras.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
#Returns A list in which each item is a dictionary with details about an input tensor. Each dictionary contains the following fields that describe the tensor:
#name: The tensor name.
#index: The tensor index in the interpreter.
#shape: The shape of the tensor.
#shape_signature: Same as shape for models with known/fixed shapes. If any dimension sizes are unkown, they are indicated with -1.
#dtype: The numpy data type (such as np.int32 or np.uint8).
#quantization: Deprecated, use quantization_parameters. This field only works for per-tensor quantization, whereas quantization_parameters works in all cases.
#quantization_parameters: A dictionary of parameters used to quantize the tensor: ~ scales: List of scales (one if per-tensor quantization). ~ zero_points: List of zero_points (one if per-tensor quantization). ~ quantized_dimension: Specifies the dimension of per-axis quantization, in the case of multiple scales/zero_points.
#sparsity_parameters: A dictionary of parameters used to encode a sparse tensor. This is empty if the tensor is dense.
output_details = interpreter.get_output_details() # Gets model output tensor details.
#Returns A list in which each item is a dictionary with details
# about an output tensor. The dictionary contains the same fields as described for get_input_details().

# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
#index: The tensor index in the interpreter.
interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print("input data: " , input_data,"output data:" , output_data)


# In[56]:


model.predict([0.15172438])


# In[39]:


model.predict([1.0])


# In[ ]:





# In[70]:


#Loading the inference model

import numpy as np
import tensorflow as tf

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="saved_polynomial_TestingV2tflite_keras.tflite")
interpreter.allocate_tensors()
xs = np.array([-1.0,0.0,1.0,2.0,3.0,4.0], dtype = float)
# Get input and output tensors.
input_details = interpreter.get_input_details()
#Returns A list in which each item is a dictionary with details about an input tensor. Each dictionary contains the following fields that describe the tensor:
#name: The tensor name.
#index: The tensor index in the interpreter.
#shape: The shape of the tensor.
#shape_signature: Same as shape for models with known/fixed shapes. If any dimension sizes are unkown, they are indicated with -1.
#dtype: The numpy data type (such as np.int32 or np.uint8).
#quantization: Deprecated, use quantization_parameters. This field only works for per-tensor quantization, whereas quantization_parameters works in all cases.
#quantization_parameters: A dictionary of parameters used to quantize the tensor: ~ scales: List of scales (one if per-tensor quantization). ~ zero_points: List of zero_points (one if per-tensor quantization). ~ quantized_dimension: Specifies the dimension of per-axis quantization, in the case of multiple scales/zero_points.
#sparsity_parameters: A dictionary of parameters used to encode a sparse tensor. This is empty if the tensor is dense.
output_details = interpreter.get_output_details() # Gets model output tensor details.
#Returns A list in which each item is a dictionary with details
# about an output tensor. The dictionary contains the same fields as described for get_input_details().

# Test model on random input data.
input_shape = input_details[0]['shape']
#input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
input_data = np.array(np.random.random_sample(input_shape)*0 + 8.0, dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
#index: The tensor index in the interpreter.
interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print("input data: " , input_data,"output data:" , output_data)


# In[71]:


model.predict([8.0])


# In[72]:


def run_inference_polynomial_TestingV2(topredict):
    import numpy as np
    import tensorflow as tf
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="saved_polynomial_TestingV2tflite_keras.tflite")
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    #Returns A list in which each item is a dictionary with details about an input tensor. Each dictionary contains the following fields that describe the tensor:
    #name: The tensor name.
    #index: The tensor index in the interpreter.
    #shape: The shape of the tensor.
    #shape_signature: Same as shape for models with known/fixed shapes. If any dimension sizes are unkown, they are indicated with -1.
    #dtype: The numpy data type (such as np.int32 or np.uint8).
    #quantization: Deprecated, use quantization_parameters. This field only works for per-tensor quantization, whereas quantization_parameters works in all cases.
    #quantization_parameters: A dictionary of parameters used to quantize the tensor: ~ scales: List of scales (one if per-tensor quantization). ~ zero_points: List of zero_points (one if per-tensor quantization). ~ quantized_dimension: Specifies the dimension of per-axis quantization, in the case of multiple scales/zero_points.
    #sparsity_parameters: A dictionary of parameters used to encode a sparse tensor. This is empty if the tensor is dense.
    output_details = interpreter.get_output_details() # Gets model output tensor details.
    #Returns A list in which each item is a dictionary with details
    # about an output tensor. The dictionary contains the same fields as described for get_input_details().

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    #input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    input_data = np.array(np.random.random_sample(input_shape)*0 + topredict, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    #index: The tensor index in the interpreter.
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("input data: " , input_data,"output data:" , output_data)


# In[73]:


run_inference_polynomial_TestingV2(7.0)


# In[ ]:





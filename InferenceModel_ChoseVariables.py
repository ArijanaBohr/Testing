#!/usr/bin/env python
# coding: utf-8

# In[1]:
import sys

def run_inference_polynomial_TestingV2(topredict):
    import numpy as np
    import tflite_runtime.interpreter as tflite
    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path="saved_polynomial_TestingV2tflite_keras.tflite")
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
    #input_data = np.array(np.random.random_sample(input_shape), dtype)
    input_data = np.array(np.random.random_sample(input_shape)*0 + float(topredict), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    #index: The tensor index in the interpreter.
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("input data: " , input_data,"output data:" , output_data)
#example: run_inference_polynomial_TestingV2(8.0)
if __name__ == '__main__':
    # Map command line arguments to function arguments.
    run_inference_polynomial_TestingV2(sys.argv[1])

# In[ ]:





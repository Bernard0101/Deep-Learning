import os
import sys


funzioni_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Funzioni'))

sys.path.append(funzioni_path)


#aggiornamento dell'import dello dataset
from mnist import MNIST

# Carica i dati MNIST
mndata = MNIST('data/')
X_train, y_train = mndata.load_training


import numpy as np 
import matplotlib as plt
import functions as CNN_func


class ConvolutionalNN_Structure:

    def __init__(self, features, CNN_layers=[]):
        self.parameters=[]
        self.CNN_layers=CNN_layers
        self.ativazioni=[]

    def inputLayer(self, features):
        input_weights=self.weights["input"]["weights"]
        input_bias=self.bias["input"]["bias"]
        result=np.dot(features, input_weights.T) + input_bias
        return result
    
    def hiddenLayer(self, features, layer):
        hidden_weights=self.weights["hidden"][layer]["weights"]
        hidden_bias=self.bias["hidden"][layer]["bias"]
        result=np.dot(features, hidden_weights.T) + hidden_bias
        return result
    
    def outputLayer(self, features):
        output_weights=self.weights["output"][-1]["weights"]
        output_bias=self.bias["output"][-1]["bias"]
        result=np.dot(features, output_weights.T) + output_bias
        return result

    def ConvolutionalLayer(self, features):
        pass

    def PoolingLayer(self, feature_map):
        pass


    def initialize_parameters(self):
        for i, layer in enumerate(self.CNN_layers):

            layer_key = f"layer_{i+1}"

            if layer["type"] == "input":
                input_neurons=layer["neurons"]
                input_weights=layer["weights"]
                self.parameters[layer_key] = {
                    "layer" : "input",
                    "weights" : np.random.randn(input_weights, input_neurons),
                    "bias" : np.random.randn(input_neurons)
                }

            elif layer["type"] == "hidden":
                hidden_neurons=layer["neurons"]
                prev_neurons=self.CNN_layers[i-1]["neurons"] 
                self.parameters[layer_key] = {
                    "layer" : "hidden",
                    "weights" : np.random.randn(prev_neurons, hidden_neurons),
                    "bias" : np.random.randn(hidden_neurons)
                }

            elif layer["type"] == "output":
                output_neurons=layer["neurons"]
                prev_neurons=self.CNN_layers[i-1]["neurons"]
                self.parameters[layer_key]={
                    "layer" : "output",
                    "weights" : np.random.randn(prev_neurons, output_neurons),
                    "bias" : np.random.randn(output_neurons)
                }

            elif layer["type"] == "conv":
                pass

            elif layer["type"] == "pool":
                pass


CNN_struct = ConvolutionalNN_Structure(features=None CNN_layers=[{
                                    "type" : "input", "neurons" : features,
                                    "type" : "conv", "kernel_size" : (3,3), "filters" : 4,
                                    "type" : "conv", "kernel_size" : (3,3), "filters" : 4,
                                    "type" : "conv", "kernel_size" : (3,3), "filters" : 4,
                                    "type" : "pool",
                                    "type" : "hidden", "neurons" : 12,
                                    "type" : "hidden", "neurons" : 24,
                                    "type" : "hidden", "neurons" : 12,
                                    "type" : "output", "neurons" : 1
                                    }])



class ConvolutionalNN_Architecture:
    def __init__(self):
        pass
        

    def Forward():

                
    

 
   
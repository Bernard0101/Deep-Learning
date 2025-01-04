import os
import sys


funzioni_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Funzioni'))

sys.path.append(funzioni_path)


#aggiornamento dell'import dello dataset
from mnist import MNIST # type: ignore

# Carica i dati MNIST
mndata = MNIST('data/')
features, labels = mndata.load_training


import numpy as np  # type: ignore
import matplotlib as plt # type: ignore
import functions as CNN_func # type: ignore


class ConvolutionalNN_Structure:

    def __init__(self, features=features, CNN_layers=[]):
        self.parameters={}
        self.CNN_layers=CNN_layers
        self.ativazioni=[]

    #il metodo per il strato d'input
    def inputLayer(self, features):
        if not self.parameters[0]["type"] == "input":
            raise ValueError("tipo de layer scoretto")
        else:
            input_weights=self.parameters[0]["weights"]
            input_bias=self.parameters[0]["bias"]
            result=np.dot(features, input_weights.T) + input_bias
            return result
    
    #il metodo per I strati nacosti
    def hiddenLayer(self, features, layer):
        if not self.parameters[layer]["type"] == "hidden":
            raise ValueError("tipo de layer scoretto")
        else:
            hidden_weights=self.parameters[layer]["weights"]
            hidden_bias=self.parameters[layer]["bias"]
            result=np.dot(features, hidden_weights.T) + hidden_bias
            return result
    
    #il metodo per il starto d'uscita
    def outputLayer(self, features):
        if not self.parameters[-1]["type"] == "output":
            raise ValueError("tipo de layer scorreto")
        else:
            output_weights=self.parameters[-1]["weights"]
            output_bias=self.parameters[-1]["bias"]
            result=np.dot(features, output_weights.T) + output_bias
            return result

    #il metodo per il strato convoluzionale
    def ConvolutionalLayer(self, features, layer):
        pass


    #il metodo per il strato di pooling
    def PoolingLayer(self, feature_map):
        pass

    #il metodo che initializa tutti strati della architettura datta in CNN_layers
    def initialize_parameters(self):
        for i, layer in enumerate(self.CNN_layers):

            layer_key = f"layer_{i}"

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
                kernel_length = layer["kernel_length"]
                kenrnel_height = layer["kernel_height"]
                filters = layer["filters"]
                self.parameters[layer_key]={
                    "layer" : "conv",
                    "kernels" : np.random.randn(kenrnel_height, kernel_length , filters) 
                }

            elif layer["type"] == "pool":
                pass


CNN_struct = ConvolutionalNN_Structure(features=None, CNN_layers=[
                                    {"type" : "input", "neurons" : features},
                                    {"type" : "conv", "kernel_height" : 3, "kernel_length" : 3, "filters" : 4},
                                    {"type" : "conv", "kernel_height" : 3, "kernel_length" : 3, "filters" : 4},
                                    {"type" : "conv", "kernel_height" : 3, "kernel_length" : 3, "filters" : 4},
                                    {"type" : "pool"},
                                    {"type" : "hidden", "neurons" : 12},
                                    {"type" : "hidden", "neurons" : 24},
                                    {"type" : "hidden", "neurons" : 12},
                                    {"type" : "output", "neurons" : 1}
                                    ])

                
    

 
   
import os
import sys


# Aggiungi la cartella Funzioni al sistema dei percorsi
funzioni_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Funzioni'))
sys.path.append(funzioni_path)


import tensorflow as tf

#Scarica automaticamente CIFAR-10
(features, _), _ = tf.keras.datasets.cifar10.load_data()
input_size = features.shape[1] * features.shape[2] * features.shape[3]
print("Dimensione di input per una singola immagine:", input_size)



import numpy as np
import matplotlib as plt
import functions as CNN_func


class ConvolutionalNN_Structure:

    def __init__(self, features=None, CNN_layers=[]):
        self.parameters={}
        self.CNN_layers=CNN_layers
        self.ativazioni=[]

    #il metodo per il strato d'input
    def inputLayer(self, features):
        if self.parameters[0]["type"] != "input":
            raise ValueError("tipo di strato scoretto")
        else:
            input_weights=self.parameters[0]["weights"]
            input_bias=self.parameters[0]["bias"]
            result=np.dot(features, input_weights.T) + input_bias
            return result
    
    #il metodo per I strati nacosti
    def hiddenLayer(self, features, layer):
        if self.parameters[layer]["type"] != "hidden":
            raise ValueError("tipo di strato scoretto")
        else:
            hidden_weights=self.parameters[layer]["weights"]
            hidden_bias=self.parameters[layer]["bias"]
            result=np.dot(features, hidden_weights.T) + hidden_bias
            return result
    
    #il metodo per il starto d'uscita
    def outputLayer(self, features):
        if self.parameters[-1]["type"] != "output":
            raise ValueError("tipo di strato scorreto")
        else:
            output_weights=self.parameters[-1]["weights"]
            output_bias=self.parameters[-1]["bias"]
            result=np.dot(features, output_weights.T) + output_bias
            return result

    #il metodo per il strato convoluzionale
    def ConvolutionalLayer(self, features, layer):
        if self.parameters[layer]["type"] != "conv":
            raise ValueError("tipo di strato scoretto")
        else:
            #prendendo le carateristiche dei kernel 
            kernels=self.parameters[layer]["kernels"]
            kernel_height=self.parameters[layer]["height"]
            kernel_length=self.parameters[layer]["length"]

            #applicando un padding all'immagine originale
            feature_map=np.zeros(features)

            #algoritmo per la convoluzione
            for i in range(features.shape[0]):
                for j in range(features.shape[1]):

                    region=features[i:i+kernel_length, j:j+kernel_height]

                    feature_map[i,j]=np.sum(region * kernels)

            return feature_map

            


    #il metodo per il strato di pooling
    def PoolingLayer(self, feature_map):
        pass

    #il metodo che initializa tutti strati della architettura datta in CNN_layers
    def initialize_parameters(self):
        for i, layer in enumerate(self.CNN_layers):

            layer_key = f"layer_{i}"

            if layer["type"] == "input":
                input_neurons=layer["neurons"]
                input_features=layer["feature_inputs"]
                self.parameters[layer_key] = {
                    "layer" : "input",
                    "weights" : np.random.randn(input_neurons, input_features),
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
                kernel_height = layer["kernel_height"]
                channels = 3
                self.parameters[layer_key]={
                    "layer" : "conv",
                    "height" : kernel_height,
                    "length" : kernel_length,
                    "kernels" : np.random.randn(kernel_height, kernel_length , channels) 
                }


CNN_struct = ConvolutionalNN_Structure(features=features, CNN_layers=[
                                    {"type" : "input", "feature_inputs" : input_size, "neurons" : 20},
                                    {"type" : "conv", "kernel_height" : 3, "kernel_length" : 3},
                                    {"type" : "conv", "kernel_height" : 3, "kernel_length" : 3},
                                    {"type" : "conv", "kernel_height" : 3, "kernel_length" : 3},
                                    {"type" : "pool"},
                                    {"type" : "hidden", "neurons" : 12},
                                    {"type" : "hidden", "neurons" : 24},
                                    {"type" : "hidden", "neurons" : 12},
                                    {"type" : "output", "neurons" : 1}
                                    ])

class CovolutionalNN_Architecture:
    def __init__(self):
        self.data_features = features

    def Forward(self):
        CNN_struct.initialize_parameters()
        out_features=CNN_struct.inputLayer(features=self.data_features)
        CNN_struct.ativazioni.append(out_features)

        out_feature_map=CNN_struct.ConvolutionalLayer(features=out_features, layer=1)
        out_feature_map=CNN_func.activation_ReLU(Z=out_feature_map)
        CNN_struct.ativazioni.append(out_feature_map)

        out_feature_map=CNN_struct.ConvolutionalLayer(features=out_feature_map, layer=2)
        out_feature_map=CNN_func.activation_ReLU(Z=out_feature_map)
        CNN_struct.ativazioni.append(out_feature_map)

        out_feature_map=CNN_struct.ConvolutionalLayer(features=out_feature_map, layer=3)
        out_feature_map=CNN_func.activation_ReLU(Z=out_feature_map)
        CNN_struct.ativazioni.append(out_feature_map)

        out_features=CNN_struct.PoolingLayer(feature_map=out_feature_map)
        CNN_struct.ativazioni.append(out_features)

        out_features=CNN_struct.hiddenLayer(features=out_features, layer=5)
        out_features=CNN_func.activation_ReLU(Z=out_features)
        CNN_struct.ativazioni.append(out_features)

        out_features=CNN_struct.hiddenLayer(features=out_features, layer=6)
        out_features=CNN_func.activation_ReLU(Z=out_features)
        CNN_struct.ativazioni.append(out_features)

        out_features=CNN_struct.hiddenLayer(features=out_features, layer=7)
        out_features=CNN_func.activation_ReLU(Z=out_features)
        CNN_struct.ativazioni.append(out_features)

        out_features=CNN_struct.outputLayer(features=out_features)
        return out_features



        
CNN_Arc=CovolutionalNN_Architecture()
CNN_Arc.Forward()
    

 
   
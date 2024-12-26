import os
import sys

funzioni_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Funzioni'))

sys.path.append(funzioni_path)


from PIL import Image
import numpy as np 
import matplotlib as plt
import functions as CNN_func



class ConvolutionalNN_Structure:

    def __init__(self, CNN_layers={}):
        self.weights={}
        self.bias=np.random.randint()
        self.CNN_layers=CNN_layers
        self.ativazioni=[]

    def initParameters(self):
        for i, layer in enumerate(self.CNN_layers):
            layer_key = f"layer_{i+1}"
            
            if layer["type"] == "conv":
                filters = layer["filters"]
                kernel_size = layer["kernel_size"]
                self.weights[layer_key] = {
                    "type" : "conv",
                    "weights" : np.random.randn(filters, kernel_size[0], kernel_size[1]),
                    "bias" : np.zeros(filters),
                }
            elif layer["type"] == "hidden":
                neurons = layer["neurons"]
                prev_neurons = self.CNN_layers[i-1]["neurons"] if self.CNN_layers[i-1]["type"] == "input" or "hidden" else 0
                self.weights[layer_key] = {
                    "type" : "hidden",
                    "weights" : np.random.randn(prev_neurons, neurons),
                    "bias" : np.zeros(neurons)
                }
            elif layer == ["input"]:
                neurons = layer["neurons"]
                self.weights[layer_key] = {
                    "type" : "input",
                    "weights" : np.random(features, neurons)
                }


            
    #first layer of the CNN
    def inputLayer(self, in_features):
        input_weights=self.weights["input"]["weights"]
        input_bias=self.weights["input"]["bias"]
        out_features=np.dot(in_features, input_weights.T) + input_bias
        return out_features
    
    #the middle layer of the CNN
    def hiddenLayer(self, in_features, layer):
        layer_key=f"layer_{layer + 1}" 
        hidden_weights=self.weights[layer_key]["weights"]
        hidden_bias=self.weights[layer_key]["bias"]
        out_features=np.dot(in_features, hidden_weights.T) + hidden_bias
        return out_features

    #the convolutional layer of the CNN
    def ConvLayer(self, image, layer):
        kernel=self.weights[layer]

        #prende il shape e la mitta dell'altezza e larghezza dello kernel 
        kernel_Lin, kernel_Col=kernel.shape
        padLin, padCol=kernel_Lin // 2, kernel_Col // 2

        #crea un pad alla immagine
        padded_image=np.pad(image, ((padLin, padLin), (padCol, padCol)), mode="constant")
        
        #costroi la feature map
        feature_map=np.zeros(image.shape)

        #algoritmo per la convoluzione
        for i in range(image.shape[0]):
            for j in range(image.shape[1]): 
                
                #prende tutti i valori nella immagine a rispeto dello kernel
                region = padded_image[i:i+kernel_Lin, j:j+kernel_Col]

                #esegue la moltiplicazione per ogni valori e somma il risultato, aggiungindolo alla feature map
                feature_map[i, j] = np.sum(region * kernel)

        return feature_map

    def PoolingLayer(in_features, layer):
        pass

CNN_struct=ConvolutionalNN_Structure(CNN_layers=[{
                                    "type" : "input", "neurons" : Features, 
                                    "type" : "conv", "kernel_size" : (3, 3), "filters" : 5,
                                    "type" : "conv", "kernel_size" : (3, 3), "filters" : 7,
                                    "type" : "conv", "kernel_size" : (3, 3), "filters" : 4, 
                                    "type" : "pool",
                                    "type" : "hidden", "neurons" : 12, 
                                    "type" : "hidden", "neurons" : 10, 
                                    "type" : "hidden", "neurons" : 1
                                    }])

class CovolutionalNN_Architecture:
    def __init__(self):
        self.modelo = CNN_struct


        def Forward(self):
            out_features=CNN_struct.inputLayer()

            feature_map=CNN_struct.ConvLayer(image=out_features, layer=0)
            out_features=CNN_func.nn_functions.activation_ReLU(Z=feature_map)

            feature_map=CNN_struct.ConvLayer(image=out_features, layer=1)
            out_features=CNN_func.nn_functions.activation_ReLU(Z=feature_map)

            feature_map=CNN_struct.ConvLayer(image=out_features, layer=2)
            out_features=CNN_func.nn_functions.activation_ReLU(Z=feature_map)

            feature_map=CNN_struct.PoolingLayer(in_features=out_features, layer=3)
            out_features=CNN_func.nn_functions.activation_ReLU(Z=out_features)

            out_features=CNN_struct.hiddenLayer(in_features=out_features, layer=4)
            out_features=CNN_func.nn_functions.activation_ReLU(Z=out_features)

            out_features=CNN_struct.hiddenLayer(in_features=out_features, layer=5)
            out_features=CNN_func.nn_functions.activation_ReLU(Z=out_features)

            out_features=CNN_struct.hiddenLayer(in_features=out_features, layer=6)
            out_features=CNN_func.nn_functions.activation_ReLU(Z=out_features)
        
        def CalculateLoss(self, prediction):
            pass
            
                
            



                    
                        
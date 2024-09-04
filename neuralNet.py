from functions import nn_functions as nn_func
from perceptron import Perceptron, TrainPerceptron
import numpy as np
import pandas as pd

#the dataset and it`s preprocessing
data = pd.read_csv('salary_data_with_noise.csv')
features = data[['Age', 'Education']].values
labels = data['Salary'].values

p=Perceptron()
perceptron=TrainPerceptron(model=p, features=features, labels=labels)

class neuralNet_architecture():
    def __init__(self, n_features=len(features), hidden_layers=1, input_nodes=1, hidden_nodes=[1]):
        self.hidden_layers=hidden_layers
        self.input_nodes=input_nodes
        self.hidden_nodes=hidden_nodes
        self.input_weights=np.random.randn(n_features, input_nodes)
        self.hidden_weights=np.array([np.random.randn(hidden_nodes[i], hidden_nodes[i-1]) * 0.01 for i in range(1, len(hidden_nodes))], dtype=object)
        self.output_weights=np.random.randn(n_features, input_nodes)
        #self.bias=np.random.randn(layers, hidden_nodes[i])


    def input_layer(self):
        out_features=[]
        for node in range(self.input_nodes):
            Z=perceptron.predict()
            out_features.append(Z)
        return out_features


    def hidden_layer(self, x, layer):
        out_hidden_features=[]
        node_weights=self.hidden_weights[layer-1]
        for node in range(self.hidden_nodes[layer]):
            weights=node_weights[node]
            Z=0
            print(f"hidden_weights: {weights}, x: {x}")
            Z=np.dot(x, weights)
            out_hidden_features.append(Z)
        return out_hidden_features


    def output_layer(self, x):
        for preds in range(n_features):
            pass

    def activation_layer(self, x, activation="ReLU"):
        result=[]
        for preds in x:
            if(activation=="ReLU"):
                Z=nn_func.activation_ReLU(Z=preds)
                result.append(Z)
            if(activation=="leaky_ReLU"):
                Z=nn_func.activation_leaky_ReLU(Z=preds)
                result.append(Z)
        return result

    def Calculate_Loss(self):
        

            

    

nn_arc=neuralNet_architecture(hidden_layers=2, input_nodes=2, hidden_nodes=[3, 6, 2, 4])
print(f"hidden_nodes: \n\n {nn_arc.hidden_nodes}\n\n")
print(f"input weights: \n\n {nn_arc.input_weights}\n  type: {type(nn_arc.input_weights)}\n\n")
print(f"hidden_weights: \n\n {nn_arc.hidden_weights}\n  type: {type(nn_arc.hidden_weights)}\n\n")


class neuralNet():
    def __init__(self):
        pass

    def Forward(self):
        in_features=nn_arc.input_layer()
        in_features=nn_arc.hidden_layer(x=in_features, layer=0)
        in_features=nn_arc.activation_layer(x=in_features, activation="ReLU")
        print(f"\n\nin features: {in_features}")
        in_features=nn_arc.hidden_layer(x=in_features, layer=1)
        in_features=nn_arc.activation_layer(x=in_features, activation="leaky_ReLU")
        print(f"\n\nin features: {in_features}")
        in_features=nn_arc.hidden_layer(x=in_features, layer=2)
        in_features=nn_arc.activation_layer(x=in_features, activation="ReLU")
        print(f"\n\nin features: {in_features}")

    


nn=neuralNet()
nn.Forward()



        
            
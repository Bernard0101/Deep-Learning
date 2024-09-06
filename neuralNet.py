from functions import nn_functions as nn_func
from perceptron import Perceptron, TrainPerceptron
import numpy as np
import pandas as pd

#the dataset and its preprocessing
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
        self.output=[]
        self.input_weights=np.random.randn(n_features, input_nodes)
        self.hidden_weights=np.array([np.random.randn(hidden_nodes[i], hidden_nodes[i-1]) for i in range(1, len(hidden_nodes))], dtype=object)
        self.ativazzioni=[]
        #self.bias=np.random.randn(layers, hidden_nodes[i])


    def input_layer(self):
        out_features=[]
        for node in range(self.hidden_nodes[0]):
            preds=perceptron.predict()
            out_features.append(preds)
        return out_features


    def hidden_layer(self, batch, layer):
        out_hidden_features=[]
        node_weights=self.hidden_weights[layer-1]
        for node in range(self.hidden_nodes[layer]):
            weights=node_weights[node]
            Z=0
            #print(f"hidden_weights: {weights}, x: {batch}")
            Z=np.dot(batch, weights)
            out_hidden_features.append(Z)
        return out_hidden_features


    def output_layer(self, batch):
        for output in batch:
            self.output.append(output)
        return batch

            

    def activation_layer(self, batch, activation="ReLU"):
        result=[]
        for preds in batch:
            if(activation=="ReLU"):
                Z=nn_func.activation_ReLU(Z=preds)
                result.append(Z)
            if(activation=="leaky_ReLU"):
                Z=nn_func.activation_leaky_ReLU(Z=preds)
                result.append(Z)
        return result

    

            

    

nn_arc=neuralNet_architecture(hidden_layers=2, input_nodes=2, hidden_nodes=[3, 6, 3, 1])
print(f"hidden_nodes: \n\n {nn_arc.hidden_nodes}\n\n")
print(f"input weights: \n\n {nn_arc.input_weights}\n  type: {type(nn_arc.input_weights)}\n\n size: {nn_arc.input_weights.size}")
print(f"hidden_weights: \n\n {nn_arc.hidden_weights}\n  type: {type(nn_arc.hidden_weights)}\n\n size: {nn_arc.hidden_weights.size}")

class neuralNet():
    def __init__(self):
        pass

    def Forward(self):
        in_features=nn_arc.input_layer()
        in_features=nn_arc.hidden_layer(batch=in_features, layer=1)
        in_features=nn_arc.activation_layer(batch=in_features, activation="ReLU")
        nn_arc.ativazzioni.append(in_features)
        in_features=nn_arc.hidden_layer(batch=in_features, layer=2)
        in_features=nn_arc.activation_layer(batch=in_features, activation="leaky_ReLU")
        nn_arc.ativazzioni.append(in_features)
        in_features=nn_arc.hidden_layer(batch=in_features, layer=3)
        in_features=nn_arc.activation_layer(batch=in_features, activation="leaky_ReLU")


        out_features=nn_arc.output_layer(batch=in_features)
        return out_features


    def Calculate_Loss(self, target, prediction, function):
        if(function=="MSE"):
            MSE_Loss=nn_func.Loss_MSE(y_pred=prediction, y_label=target)
            return MSE_Loss
        if (function=="MAE"):
            MAE_Loss=nn_func.Loss_MAE(y_pred=prediction, y_label=target)
            return MAE_Loss


    def Backward(self, prediction, target, learning_rate=0.01):
        print("Backpropagation")
        derivata_perdita=nn_func.Loss_MSE_derivative(y_pred=nn_arc.output, y_label=target)
        derivata_ativazzione=nn_func.activation_leaky_ReLU_derivative(Z=nn_arc.output[0])
        gradiente_discendente_output=derivata_ativazzione*derivata_perdita
        for layer in reversed(range(len(nn_arc.hidden_nodes)-1)):
            print(f"\n\nlayer:{layer}")
            previous_layer=nn_arc.ativazzioni[layer-1]
            print(f"\n\ntransposed: {previous_layer} gradiente: {gradiente_discendente_output}")
            grad_W=np.dot(previous_layer, gradiente_discendente_output)
            transposed_grad_W=grad_W.T
            #print(f" weights:{nn_arc.hidden_weights[layer]}grad_W: {grad_W.T}, learning_rate: {learning_rate}")
            nn_arc.hidden_weights[layer] -= transposed_grad_W * learning_rate
            if layer > 0:
                derivata_ativazzione_hidden = nn_func.activation_leaky_ReLU_derivative(Z=previous_layer[0])
                gradiente_discendente_output = np.dot(gradiente_discendente_output, nn_arc.hidden_weights[layer].T) * derivata_ativazzione_hidden

            


   
    # Step 1: Derivative of loss with respect to predictions (output)
    
    # Step 2: Derivative of the activation function (e.g., Leaky ReLU)
    
    # Step 3: Backpropagate the error (chain rule)
    
    # Step 4: Gradient with respect to weights (for output layer)
    
    # Step 5: Update the weights (for output layer)
    #self.output_weights -= learning_rate * grad_W_output

    # Repeat similar steps for hidden layers...


        
    


    


nn=neuralNet()
predictions=nn.Forward()
Loss=nn.Calculate_Loss(prediction=predictions, target=labels, function="MSE")
print(Loss)
Backward=nn.Backward(prediction=predictions, target=labels)
print(f"new weights: {nn_arc.hidden_weights}")


from perceptron import Perceptron, TrainPerceptron
import numpy as np
import pandas as pd

#the dataset and it`s preprocessing
data = pd.read_csv('salary_data_with_noise.csv')
features = data[['Age', 'Education']].values
labels = data['Salary'].values

#perceptron objects 
p=Perceptron(learning_rate=00.1, n_features=features.shape[1])

#perceptron training objects
pt=TrainPerceptron(model=p, features=features, labels=labels)


def activation_leaky_ReLU_derivative(Z, alpha=0.01):
    return alpha if Z < 0 else 1

def activation_leaky_ReLU(z, alpha=0.01):
    return np.where(z >= 0, z, alpha * z)

def MSE_Loss_derivative(y_pred, y_label):
    return y_pred-y_label

def MSE_Loss(losses, mse_error=0):
    for  error in losses:
        mse_error+= error**2
    mse_error/=len(losses)
    return mse_error

class MultilayerPerceptron():
    def __init__(self, n_features=len(features), input_nodes=2, hidden_layers=2, hidden_nodes=3):
        self.input_nodes=input_nodes
        self.hidden_layers=hidden_layers
        self.hidden_nodes=hidden_nodes
        self.input_weights=np.random.rand(n_features, input_nodes)
        self.hidden_weights_external=np.random.rand(hidden_nodes, input_nodes)
        self.hidden_weights_internal=np.random.rand(hidden_layers, hidden_nodes, hidden_nodes)
        self.biases=[np.random.rand(hidden_nodes) for j in range(hidden_layers)] + [np.random.rand(1)]
        

    #input node, that returns a list of predictions with the size of input nodes
    def input_layer(self, input_nodes=1, out_features=[]):
        for i in range(input_nodes):
            predictions=pt.predict()
            out_features.append(predictions)
        return out_features
            

    #hidden layer, that returns a list of the new predictions with the size of hidden nodes
    def hidden_layer(self, predictions, hidden_nodes=3, layer=0):
        out_hidden_features=[]
        for node in range(hidden_nodes):
            Z=0
            if layer == 0:
                #print(f"\nlayer:{layer}\nnode:{node}\npredictions: \n{predictions},\nweights: \n{self.hidden_weights_external[node]}")
                Z=np.dot(predictions, self.hidden_weights_external[node])
            else:
                #print(f"\nlayer:{layer}\nnode:{node}\npredictions: \n{predictions},\nweights: \n{self.hidden_weights_internal[layer][node]}")
                Z=np.dot(predictions, self.hidden_weights_internal[layer][:, node])
            Z+=self.biases[-1]
            print(f"{Z}\n\n")
            out_hidden_features.append(Z)
        out_hidden_features=np.squeeze(out_hidden_features)
        return out_hidden_features


    def output_layer(self, predictions):
        sum(predictions)

    

    #the activation layer of the neural network
    def activation_layer(self, out_hidden_features, p=[]):
        for preds in out_hidden_features:
            p.append(activation_leaky_ReLU(preds))
            predictions=np.squeeze(p)
        return predictions
    

    #the calculation of the mse loss 
    def Calculate_Loss(self, predictions, errors=[], mse_Loss=0):
        for label, pred in zip(labels, predictions):
            error=pred-label
            errors.append(error)
        mse_Loss=MSE_Loss(losses=errors)
        return mse_Loss


    def Forward(self):
        #the initial input layer 
        #print("starting forward pass")
        in_features=nn.input_layer(input_nodes=self.input_nodes)

        #all the hidden layers and also its activation functions
        #print(f"\ninput_layer: \n{in_features}")
        in_features=nn.hidden_layer(predictions=in_features, hidden_nodes=self.hidden_nodes, layer=0)
        #print(f"\nlayer 0: \n{in_features}")
        in_features=nn.activation_layer(out_hidden_features=in_features)
        #print(f"\nlayer 1: \n{in_features}")
        in_features=nn.hidden_layer(predictions=in_features, hidden_nodes=self.hidden_nodes, layer=1)
        out_features=nn.activation_layer(out_hidden_features=in_features)

        #the final prediction
        print(f"the final preds ff{out_features}")
        return out_features


    #the backpropagation algorithm and its gradient descent calcculation
    def Backpropagation(self, predictions):
        derivata_perdita=MSE_Loss_derivative(y_pred=predictions, y_label=labels)
        derivata_ativazzione=activation_leaky_ReLU_derivative(Z=predictions)
        gradiente_errori=derivata_perdita*derivata_ativazzione

        






nn=MultilayerPerceptron(input_nodes=2, hidden_layers=2, hidden_nodes=3)
#print(f"{nn.input_weights}\n\n")
#print(f"{nn.hidden_weights_external}\n\n")
#print(f"{nn.hidden_weights_internal}\n\n")

class TrainMLP():
    def __init__(self, model, epochs=10):
        self.model=model
        self.epochs=epochs
    
    def training(self, losses=[]):
        #feedforward
        y_pred=nn.Forward()
        print(f"final predictions: {y_pred}")

        #calculate the loss
        mse_Loss=nn.Calculate_Loss(predictions=y_pred)
        print(f"mse Loss: {mse_Loss}")


        nn.Backpropagation(predictions=y_pred)
        


nn_train=TrainMLP(epochs=2, model=nn)
nn_train.training()
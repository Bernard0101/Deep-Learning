from perceptron import Perceptron, TrainPerceptron
import numpy as np
import pandas as pd

data = pd.read_csv('salary_data_with_noise.csv')
features = data[['Age', 'Education']].values
labels = data['Salary'].values

#perceptron objects 
p=Perceptron(learning_rate=00.1, n_features=features.shape[1])

#perceptron training objects
pt=TrainPerceptron(model=p, features=features, labels=labels)

def activation_leaky_ReLU(z, alpha=0.01):
    return np.where(z >= 0, z, alpha * z)

def MSE_Loss_derivative(alpha=0.01):
    2*()

def calculate_MSE(losses, mse_error=0):
    for index, error in enumerate(losses):
        mse_error+= error**2
    mse_error/=len(losses)
    return mse_error

class MultilayerPerceptron():
    def __init__(self, input_nodes=1, hidden_layers=1, hidden_nodes=8):
        self.input_nodes=input_nodes
        self.hidden_layers=hidden_layers
        self.input_weights=np.random.rand(len(features)*input_nodes)
        self.hidden_weights=np.random.rand(hidden_nodes*hidden_layers*features.shape[1])


    #the initial input layer with two nodes each with weights eqauling to the number 
    #of features needed to be predicted and outputing two number that will feed the next layer
    def input_layer(self, input_nodes=2, out_features=[]):
        for node in range(input_nodes):
            predictions=pt.predict()
            out_features.append(predictions)
        return out_features
            
    #hidden layer, with inputs of the previous two layers, outputing 
    def hidden_layer(self, predictions, hidden_nodes, out_hidden_features=[]):
        bias=np.random.randint(7)
        for node in range(hidden_nodes):
            Z=0
            for index, preds in enumerate(predictions):
                Z+=preds*self.hidden_weights
            Z+=bias
            out_hidden_features.append(Z)
        out_hidden_features=np.squeeze(out_hidden_features)
        return out_hidden_features
    
    #the activation layer of the neural network
    def activation_layer(self, out_hidden_features, predictions=[]):
        bias=np.random.randint(7)
        for preds in out_hidden_features:
            predictions.extend(activation_leaky_ReLU(preds))
        return predictions
    
    #the calculation of the mse loss 
    def CalculateLoss(self, predictions, errors=[], mse_loss=0):
        for label, pred in zip(labels, predictions):
            error=pred-label
            errors.append(error)
        mse_loss=calculate_MSE(losses=errors)
        return mse_loss

    #the backpropagation algorithm and its gradient descent calculus
 #   def Backpropagation(self, predictions):
        






nn=MultilayerPerceptron(input_nodes=2, hidden_layers=2)

class TrainMLP():
    def __init__(self, model, epochs=10):
        self.model=model
        self.epochs=epochs
    
    def training(self, losses=[]):
        for epoch in range(self.epochs):

            #returns an array with the initial predictions that is the amount of features * the amount of nodes
            in_features=nn.input_layer(input_nodes=nn.input_nodes)
           # print(in_features)

            #receives those in_features that are an array and passes it to the next layer trough all of the nodes in it
            for layer in range(nn.hidden_layers):
                in_features=nn.hidden_layer(predictions=in_features, hidden_nodes=3)

            #receives the predictions from the hidden layers and activate using the activation function
            in_activation_features=nn.activation_layer(out_hidden_features=in_features)
            mse_Loss=nn.CalculateLoss(predictions=in_activation_features)
            #print(mse_Loss)

nn_train=TrainMLP(epochs=10, model=nn)
nn_train.training()

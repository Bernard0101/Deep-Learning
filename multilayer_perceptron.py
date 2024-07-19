from perceptron import Perceptron, TrainPerceptron
import numpy as np
import pandas as pd

data = pd.read_csv('salary_data_with_noise.csv')
features = data[['Age', 'Education']].values
labels = data['Salary'].values


p=Perceptron(learning_rate=00.1, n_features=len(features))
pt=TrainPerceptron(p, features=features, labels=labels)

def activation_leaky_ReLU(z, alpha=0.01):
    return np.where(z >= 0, z, alpha * z)

def calculate_MSE(losses, mse_error=0):
    for index, error in enumerate(losses):
        mse_error+= error**2
    mse_error/=len(losses)
    return mse_error

class MultilayerPerceptron():
    def __init__(self):
        self.weights=np.random.rand()
        self.loss=np.random.rand()

    #the initial input layer with two nodes each with weights eqauling to the number 
    #of features needed to be predicted and outputing two number that will feed the next layer
    def input_layer(self, out_features=[]):
        pred1=pt.training_loop()
        pred2=pt.training_loop()
        out_features.append(pred1, pred2)
        return out_features

    #first hidden layer, with inputs of the previous two layers, outputing 
    def hidden_layer(self, predictions, hidden_nodes, new_preds=[]):
        self.weights=np.random.rand(len(predictions))
        self.bias=np.random.randint(7)
        for node in range(hidden_nodes):
            Z=0
            for index, preds in enumerate(predictions):
                Z+=preds*self.weights[index]
            Z+=self.bias
            new_preds.append(Z)
        return new_preds
    
    #the activation layer of the neural network
    def activation_layer(self, new_preds, predictions=[]):
        bias=np.random.randint(7)
        for preds in new_preds:
            predictions.append(activation_leaky_ReLU(preds))
        return predictions
    
    def CalculateLoss(self, predictions, errors=[], mse_loss=0):
        for label, pred in zip(labels, predictions):
            error=pred-label
            errors.append(error)
        mse_loss=calculate_MSE(losses=errors)
        return mse_loss
    
    #def Backpropagation(self):


nn=MultilayerPerceptron()

class TrainMLP():
    def __init__(self, epochs=100):
        self.epochs=epochs
    
    def training(self, losses=[]):
        for epoch in range(self.epochs):
           in_features=nn.input_layer()
           in_hidden_features=nn.hidden_layer(predictions=in_features, hidden_nodes=4)
           in_hidden_features=nn.hidden_layer(predictions=in_hidden_features, hidden_nodes=4)
           in_activation_features=nn.activation_layer(new_preds=in_hidden_features)
           mse_Loss=nn.CalculateLoss(predictions=in_activation_features)

           losses.append(mse_Loss)

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

#the derivative of leaky ReLU
def activation_leaky_ReLU_derivative(Z, alpha=0.01):
   # print(f"Z: {Z}")
    return np.where(Z > 0, 1, alpha)

#Leaky ReLU
def activation_leaky_ReLU(z, alpha=0.01):
    return np.where(z >= 0, z, alpha * z)

#the derivative of mse loss
def MSE_Loss_derivative(y_pred, y_label):
    #print(f"y_pred; {y_pred},y_label: {y_label}")
    return (2 * (y_pred-y_label) / len(y_label))

#mse Loss
def MSE_Loss(y_pred, y_label):
    mse_loss=np.mean((y_pred-y_label)**2)
    #print(f"mse_loss: {mse_loss}")
    return mse_loss

class MultilayerPerceptron():
    def __init__(self, n_features=len(features), input_nodes=2, hidden_layers=5, hidden_nodes=[3]):
        self.hidden_layers=hidden_layers
        self.input_nodes=input_nodes
        self.hidden_nodes=hidden_nodes
        self.input_weights=np.random.randn(n_features, input_nodes)
        self.hidden_weights=np.array([np.random.randn(hidden_nodes[i], hidden_nodes[i-1]) * 0.01 for i in range(1, len(hidden_nodes))], dtype=object)
        self.biases=[np.random.randn(hidden_layers) for j in range(1, len(hidden_nodes))] + [np.random.rand(1)]
        

    #input node, that returns a list of predictions with the size of input nodes
    def input_layer(self, input_nodes=1):
        out_features=[]
        for i in range(input_nodes):
            predictions=pt.predict()
            out_features.append(predictions)
        #print(f"input weights: \n{out_features}")
        return out_features
            

    #hidden layer, that returns a list of the new predictions with the size of hidden nodes
    def hidden_layer(self, predictions, hidden_nodes=3, layer=0):
        out_hidden_features=[]
        node_weights=self.hidden_weights[layer-1]
        for node in range(self.hidden_nodes[layer]):
            weights=node_weights[node]
            Z=0
            #print(f"\nlayer:{layer}\nnode:{node}\npredictions: \n{predictions},\nweights: \n{weights}")
            Z=np.dot(predictions, weights)
            Z+=self.biases[-1]
            out_hidden_features.append(Z)
        out_hidden_features=np.squeeze(out_hidden_features)
        #print(f"out_features{out_hidden_features}")
        return out_hidden_features


    

    #the activation layer of the neural network
    def activation_layer(self, out_hidden_features):
        p=[]
        #print(f"\n\nactivation weights: {out_hidden_features}")
        if np.ndim(out_hidden_features)!=0:
            for preds in out_hidden_features:
                p.append(activation_leaky_ReLU(preds))
                predictions=np.squeeze(p)
        else:
            predictions=activation_leaky_ReLU(out_hidden_features)
        #print(f"\n\nactivation predizzioni:{predictions}\n\n")
        return predictions
    

    #the calculation of the mse loss 
    def Calculate_Loss(self, predictions, mse_Loss=0):
        mse_Loss=MSE_Loss(y_pred=predictions, y_label=labels)
        return mse_Loss


    def Forward(self):
        #the initial input layer 
        #print("starting forward pass")
        in_features=nn.input_layer(input_nodes=self.input_nodes)

        #all the hidden layers and also its activation functions
        in_features=nn.hidden_layer(predictions=in_features, hidden_nodes=3, layer=1)
        in_features=nn.activation_layer(out_hidden_features=in_features)
        in_features=nn.hidden_layer(predictions=in_features, hidden_nodes=9, layer=2)
        in_features=nn.activation_layer(out_hidden_features=in_features)
        out_features=nn.hidden_layer(predictions=in_features, hidden_nodes=3, layer=3)

        return out_features

    #the backpropagation algorithm and its gradient descent calculation
    def Backpropagation(self, predictions, learning_rate):
        derivata_perdita=MSE_Loss_derivative(y_pred=predictions, y_label=labels)#array=3
        derivata_ativazione=activation_leaky_ReLU_derivative(Z=predictions)
        for idx in range(len(self.hidden_weights)-1,-1,-1):
            print(idx)
            weights=self.hidden_weights.reshape(-1, 1)
            gradiente_discendente=np.dot(weights, derivata_perdita)
            gradiente_pesi=np.dot(self.hidden_weights[idx]-gradiente_discendente*learning_rate)
            

                

            


        
nn=MultilayerPerceptron(input_nodes=2, hidden_nodes=[2, 3, 9, 3], hidden_layers=2)
#print(f"hidden_nodes: \n\n {nn.hidden_nodes}\n\n")
#print(f"input weights: \n\n {nn.input_weights}\n shape: {nn.input_weights.shape}\n type: {type(nn.input_weights)}\n\n")
#print(f"hidden_weights: \n\n {nn.hidden_weights}\n shape: {nn.hidden_weights.shape}\n type: {type(nn.hidden_weights)}\n\n")

class TrainMLP():
    def __init__(self, model, epochs=10, learning_rate=0.01):
        self.model=model
        self.epochs=epochs
        self.learning_rate=learning_rate
    
    def training(self, losses=[]):
    #for epoch in range(self.epochs):
        #feedforward
        y_pred=nn.Forward()
        print("\n\n\n\n--------------------------------------------------------")
        print(f"final predictions: {y_pred}")

        #calculate the loss
        mse_Loss=nn.Calculate_Loss(predictions=y_pred)
        print(f"mse Loss: {mse_Loss}\n\n\n")


        nn.Backpropagation(predictions=y_pred, learning_rate=self.learning_rate)
        #if epoch % 2 == 0:
        #    print(f"epoch: {epoch}| predictions: {y_pred}| loss: {mse_Loss}")
        


nn_train=TrainMLP(epochs=1, model=nn, learning_rate=0.001)
nn_train.training()
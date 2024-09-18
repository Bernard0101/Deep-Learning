from functions import nn_functions as nn_func
import numpy as np
import pandas as pd

#the dataset and its preprocessing
data = pd.read_csv('dataset.csv')
features = data[['Age', 'Education']].values
print(f"\n\nfeatures: {features}")
labels = data['Salary'].values


#the main class for creating the neural net it provides the tools for building a neuralNet architecture
class neuralNet_architecture():
    def __init__(self, n_features=len(features), hidden_layers=1, hidden_nodes=[], output_nodes=1):
        self.hidden_layers=hidden_layers
        self.hidden_nodes=hidden_nodes
        self.output_nodes=output_nodes
        self.input_weights=np.random.randn(hidden_nodes[0], n_features)
        self.hidden_weights=np.array([np.random.randn(hidden_nodes[i], hidden_nodes[i-1]) for i in range(1, len(hidden_nodes))], dtype=object)
        self.output_weights=np.random.randn(n_features, output_nodes)
        self.ativazzioni=[]
        #self.bias=np.random.randn(layers, hidden_nodes[i])

    
    def input_layer(self, batch):
        Z=np.dot(self.input_weights, batch)
        out_features=np.sum(Z, axis=1)
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
        Z=np.dot(batch, self.output_weights.T)
        return Z

            

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

    

nn_arc=neuralNet_architecture(hidden_layers=2, hidden_nodes=[2, 6, 3, 1])
print(f"hidden_nodes: \n\n {nn_arc.hidden_nodes}\n\n")
print(f"input weights: \n\n {nn_arc.input_weights}\n  type: {type(nn_arc.input_weights)}\n\n size: {nn_arc.input_weights.size}")
print(f"hidden_weights: \n\n {nn_arc.hidden_weights}\n  type: {type(nn_arc.hidden_weights)}\n\n size: {nn_arc.hidden_weights.size}")
print(f"output_weights: \n\n {nn_arc.output_weights}\n type: {type(nn_arc.output_weights)}\n\n size: {nn_arc.output_weights.size}")

#the class that contains the Forward, Backward and calculate loss methods
class neuralNet():
    def __init__(self):
        pass

    def Forward(self):
        "outputs a vector of size 2"
        in_features=nn_arc.input_layer(batch=features)

        "outputs a vector of size 6"
        in_features=nn_arc.hidden_layer(batch=in_features, layer=1)
        in_features=nn_arc.activation_layer(batch=in_features, activation="ReLU")
        nn_arc.ativazzioni.append(in_features)
        #print(f"layer 1 ativazioni: {in_features}\n")

        "outputs a vector of size 3"
        in_features=nn_arc.hidden_layer(batch=in_features, layer=2)
        in_features=nn_arc.activation_layer(batch=in_features, activation="leaky_ReLU")
        nn_arc.ativazzioni.append(in_features)
        #print(f"layer 2 ativazioni: {in_features}\n")

        "outputs a vector of size 1"
        in_features=nn_arc.hidden_layer(batch=in_features, layer=3)
        in_features=nn_arc.activation_layer(batch=in_features, activation="leaky_ReLU")
        nn_arc.ativazzioni.append(in_features)
        #print(f"layer 3 ativazioni: {in_features}\n")


        out_features=nn_arc.output_layer(batch=in_features)

        #print(f"______________________________________________________________\n\n\nfinal preds: {out_features}")
        return out_features


    def Calculate_Loss(self, target, prediction, function):
        if(function=="MSE"):
            MSE_Loss=nn_func.Loss_MSE(y_pred=prediction, y_label=target)
            return MSE_Loss
        if (function=="MAE"):
            MAE_Loss=nn_func.Loss_MAE(y_pred=prediction, y_label=target)
            return MAE_Loss


    def Backward(self, prediction, target, learning_rate=0.01):
        
        #calculo del gradiente discendente de uscita 
        derivata_perdita=nn_func.Loss_MSE_derivative(y_pred=prediction, y_label=target)
        derivata_ativazzione=nn_func.activation_leaky_ReLU_derivative(Z=prediction)
        gradiente_discendente=derivata_ativazzione*derivata_perdita

        #Update weights of the output layer 
        gradiente = gradiente_discendente*nn_arc.ativazzioni[-1][0]
        gradiente = np.sum(gradiente, axis=0, keepdims=True)
        nn_arc.output_weights -= learning_rate * gradiente


        #Backpropagate the error to hidden layers 
        for i in reversed(range(len(nn_arc.hidden_weights))):
            if i > 0:
                derivata_ativazione_hidden=nn_func.activation_ReLU_derivative(Z=np.mean(nn_arc.ativazzioni[i]))

                if (gradiente_discendente.shape[0] == nn_arc.hidden_weights[i].T.shape[0]):
                    #print(f"gradiente discendente: {gradiente_discendente.shape},\n pesi: {nn_arc.hidden_weights[i].T.shape}")
                    gradiente_hidden=np.dot(gradiente_discendente, nn_arc.hidden_weights[i].T) * derivata_ativazione_hidden
                else:
                    #print(f"gradiente discendente: {gradiente_discendente.shape},\n pesi: {nn_arc.hidden_weights[i].shape}")
                    gradiente_hidden=np.dot(gradiente_discendente, nn_arc.hidden_weights[i]) * derivata_ativazione_hidden

                #agiornare I pesi
                nn_arc.hidden_weights[i]-=gradiente_hidden*learning_rate
            else: 
                derivata_ativazione_hidden=nn_func.activation_ReLU_derivative(Z=np.mean(nn_arc.ativazzioni[i]))
                gradiente_padded=np.pad(gradiente_discendente, (0, 3), mode='constant')
                #print(f"gradiente discendente: {gradiente_padded.shape},\n pesi: {nn_arc.hidden_weights[i].shape}")
                gradiente_hidden=np.dot(gradiente_padded, nn_arc.hidden_weights[i]) * derivata_ativazione_hidden
        
        derivata_ativazione_input=nn_func.activation_ReLU_derivative(Z=np.mean(nn_arc.ativazzioni[i]))
        #print(f"gradiente input: {gradiente_hidden.shape}, input weights: {nn_arc.input_weights.shape}")
        gradiente_input=np.dot(gradiente_hidden, nn_arc.input_weights) * derivata_ativazione_input
        nn_arc.input_weights[i]-=gradiente_input*learning_rate
                
    
#the class that trains the neural Net
class trainNeuralNet():
    def __init__(self, epochs=25):
        self.epochs=epochs

    def train(self):
        for epoch in range(self.epochs):
            nn=neuralNet()

            predictions=nn.Forward()
            #print(predictions)
            Loss=nn.Calculate_Loss(prediction=predictions, target=labels, function="MSE")
            #print(Loss)
            Backward=nn.Backward(prediction=predictions, target=labels)
            #print(f"new weights: {nn_arc.hidden_weights}")

            if epoch % 2 == 0:
                print(f"epoch: {epoch}|Loss: {Loss}| predictions: {predictions}")

train=trainNeuralNet(epochs=10)
train.train()


    
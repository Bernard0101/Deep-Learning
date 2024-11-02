from functions import nn_functions as nn_func
import numpy as np
import pandas as pd

dataset=pd.read_csv("Dataset_Legge_Di_Coulomb.csv")
data_features = dataset[['Charge 1 (Coulombs)','Charge 2 (Coulombs)','Distance (m)']].values
data_labels = dataset['Force (N)'].values
data_features_cologne=data_features.shape[1]

class NeuralNetArchitecture:
    def __init__(self, features=data_features, n_features=len(data_features), nnLayers=[]):
        self.features=features
        self.n_features=n_features
        self.nnLayers=nnLayers
        self.pesi=[np.random.randn(nnLayers[0], n_features)]+[np.random.randn(nnLayers[i], nnLayers[i-1])for i in range(1, len(nnLayers))]
        self.bias=[np.random.randn(nnLayers[i])for i in range(len(nnLayers))]
        self.ativazzioni=[]

    # il metodo che raggiunge il primo layer della rette con il numero de node uguale 
    # alla quantitta di features 
    def Arc_inputLayer(self, layer=0):
        input_pesi=self.pesi[layer]
        out_features=np.dot(self.features, input_pesi.T)
        #print(f"matrice risultante del input layer: \n{out_features}\n\n")
        return out_features
    
    # gli layer nascosti dove la quantita di pesi in un node e uguale a la quantita di 
    # node nel ultimo layer 
    def Arc_hiddenLayer(self, in_features, layer):
        pesi_nascosti=self.pesi[layer]
        out_hidden_features=np.dot(in_features, pesi_nascosti.T)
        #print(f"matrice risultante del hidden layer: \n{out_hidden_features}\n\n")
        return out_hidden_features
    
    #l`ultimo layer dove se trova la predizione finalle della rette
    def Arc_outputLayer(self, in_features, layer=4):
        pesi_output=self.pesi[layer]
        out_features=np.dot(in_features, pesi_output.T)
        #print(f"matrice risulatente del output layer {out_features}\n\n")
        return out_features

nn_Arc=NeuralNetArchitecture(features=data_features, n_features=data_features_cologne, nnLayers=[4, 8, 16, 8, 1])
print(len(nn_Arc.nnLayers))
print(f"---------------------------------------------------------\nGli pesi della rette neurale: \n{nn_Arc.pesi}")
print(f"---------------------------------------------------------\nGli bias della rette neurale: \n{nn_Arc.bias}\n\n\n\n\n")

class NeuralNetwork:
    def __init__(self):
        self.predizione=None

    def Forward(self):
        out_features=nn_Arc.Arc_inputLayer(layer=0)
        nn_Arc.ativazzioni.append(out_features)

        out_features=nn_Arc.Arc_hiddenLayer(in_features=out_features, layer=1)
        out_features=nn_func.activation_ReLU(Z=out_features)
        nn_Arc.ativazzioni.append(out_features)

        out_features=nn_Arc.Arc_hiddenLayer(in_features=out_features, layer=2)
        out_features=nn_func.activation_ReLU(Z=out_features)
        nn_Arc.ativazzioni.append(out_features)

        out_features=nn_Arc.Arc_hiddenLayer(in_features=out_features, layer=3)
        out_features=nn_func.activation_ReLU(Z=out_features)
        nn_Arc.ativazzioni.append(out_features)

        out_features=nn_Arc.Arc_outputLayer(in_features=out_features)
        nn_Arc.ativazzioni.append(out_features)

        self.predizione=out_features
        #print(f"out features:{out_features}")
        return out_features


    def calculateLoss(self, target, predizione, function):
        if(function == "MSE"):
            MSE_Loss=nn_func.Loss_MSE(y_pred=predizione, y_label=target)
            return MSE_Loss
        if (function == "MAE"):
            MAE_Loss=nn_func.Loss_MAE(y_pred=predizione, y_label=target)
            return MAE_Loss


    def Backward(self, lr):
        for layer in reversed(range(len(nn_Arc.nnLayers))):
            layer_ativazioni_indietro=nn_Arc.ativazzioni[layer-1]
            layer_ativazione=nn_Arc.ativazzioni[layer]

            #derivata a rispeto della funzione di perdita
            derivata_errore=nn_func.Loss_MSE_derivative(layer_ativazioni_indietro.T, data_labels)

            #derivata a rispeto della funzione de ativazzione
            derivata_ativazione=nn_func.activation_ReLU_derivative(layer_ativazione)

            #calcolo del errore locale
            gradiente=derivata_ativazione * derivata_errore

            #adesso fare il calcolo del gradiente a rispeto di ogni pesi e bias 
            derivata_pesi=np.dot(layer_ativazioni_indietro.T, gradiente)
            derivata_bias=np.sum(gradiente, axis=0, keepdims=True)
            

            #aggiornamento dei pesi e bias
            nn_Arc.pesi[layer] -= lr * derivata_pesi.T
            nn_Arc.bias[layer] -= lr * derivata_bias.T




nn=NeuralNetwork()


class TrainNeuralNetwork():
    def __init__(self, epochs, learninig_rate):
        self.epochs=epochs
        self.lr=learninig_rate

    def train(self):
        for epoch in range(self.epochs):

            #forward pass
            preds=nn.Forward()

            #calculate Loss
            loss=nn.calculateLoss(target=data_labels, predizione=preds, function="MSE")

            #backward pass
            nn.Backward(lr=0.001)
            
            print(f"epoch: {epoch}, loss: {loss}")

train=TrainNeuralNetwork(epochs=4, learninig_rate=0.005)
train.train()


import os
import sys

# Aggiungi la cartella Funzioni al sistema dei percorsi
funzioni_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Funzioni'))
sys.path.append(funzioni_path)


import functions as nn_func
from VisualizareDatti import DatasetLeggiCoulomb
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd


dataset=DatasetLeggiCoulomb(df="Rette Neurale Multistrato/dataset_Legge_di_coulomb/Dataset_Legge_Di_Coulomb.csv")
dataset_standardized=dataset.standartizareDatti()

data_features = dataset_standardized[['Carga 1 (Coulombs)','Carga 2 (Coulombs)','Distanza (m)']].values
data_labels = dataset_standardized['Forza (N)'].values
data_features_cologne=data_features.shape[1]


class NeuralNetStructure:
    def __init__(self, n_features=len(data_features), nnLayers=[], init_pesi=None):
        self.inizializzazione=init_pesi
        self.n_features=n_features
        self.nnLayers=nnLayers
        self.pesi=[np.random.randn(nnLayers[0], n_features)] + [np.random.randn(nnLayers[i], nnLayers[i-1])for i in range(1, len(nnLayers))]
        self.bias=[np.random.randn(nnLayers[i])for i in range(len(nnLayers))] 
        self.ativazzioni=[]

    # il metodo che raggiunge il primo layer della rette con il numero de node uguale 
    # alla quantitta di features 
    def Arc_inputLayer(self, in_features, layer=0):
        input_pesi=self.pesi[layer]
        out_features=np.dot(in_features, input_pesi.T)
        #print(f"matrice risultante del input layer: \n{out_features}\n\n")
        return out_features
    
    # I starti nascosti dove la quantita di pesi in un node e uguale a la quantita di 
    # node nel ultimo layer 
    def Arc_hiddenLayer(self, in_features, layer):
        pesi_nascosti=self.pesi[layer]
        out_hidden_features=np.dot(in_features, pesi_nascosti.T)
        return out_hidden_features
    
    #l`ultimo layer dove se trova la predizione finalle della rette
    def Arc_outputLayer(self, in_features, layer=4):
        pesi_output=self.pesi[layer]
        out_features=np.dot(in_features, pesi_output.T)
        return out_features
    
    def initialize_Weights(self):

        #implementa la ativazzione Xavier
        if self.inizializzazione == "Xavier":
            Xavier_inizializzazione = np.sqrt(6 / (self.nnLayers[0] + self.nnLayers[-1]))
            self.pesi = [Xavier_inizializzazione * peso for peso in self.pesi]

        #implementa la ativazzione He
        if self.inizializzazione == "He":
            He_inizialiazzazione = np.sqrt(2 / (self.nnLayers[0]))
            self.pesi = [He_inizialiazzazione * peso for peso in self.pesi]
    

nn_Struct=NeuralNetStructure(n_features=data_features_cologne, nnLayers=[4, 8, 16, 4, 1], init_pesi="He")
print(len(nn_Struct.nnLayers))
print(f"---------------------------------------------------------\nI pesi della rette neurale: \n{nn_Struct.pesi}")
print(f"---------------------------------------------------------\nI bias della rette neurale: \n{nn_Struct.bias}\n\n\n\n\n")

class NeuralNetArchitecture:
    def __init__(self, input_data=None):
        self.predizione=None
        self.input_data=input_data

    def Forward(self):
        nn_Struct.initialize_Weights()
        out_features=nn_Struct.Arc_inputLayer(in_features=self.input_data, layer=0)
        nn_Struct.ativazzioni.append(out_features)

        out_features=nn_Struct.Arc_hiddenLayer(in_features=out_features, layer=1)
        out_features=nn_func.activation_leaky_ReLU(Z=out_features)
        nn_Struct.ativazzioni.append(out_features)

        out_features=nn_Struct.Arc_hiddenLayer(in_features=out_features, layer=2)
        out_features=nn_func.activation_leaky_ReLU(Z=out_features)
        nn_Struct.ativazzioni.append(out_features)

        out_features=nn_Struct.Arc_hiddenLayer(in_features=out_features, layer=3)
        out_features=nn_func.activation_leaky_ReLU(Z=out_features)
        nn_Struct.ativazzioni.append(out_features)

        out_features=nn_Struct.Arc_outputLayer(in_features=out_features)
        nn_Struct.ativazzioni.append(out_features)

        self.predizione=out_features
        return out_features


    def calculateLoss(self, target, predizione, function):
        if(function == "MSE"):
            MSE_Loss=nn_func.Loss_MSE(y_pred=predizione, y_label=target)
            return MSE_Loss
        if (function == "MAE"):
            MAE_Loss=nn_func.Loss_MAE(y_pred=predizione, y_label=target)
            return MAE_Loss


    def Backward(self, optim, lr=0.01):
        if(optim == "SGD"):
            nn_func.optimizer_SGD(nn_func, layers=nn_Struct.nnLayers, ativazzioni=nn_Struct.ativazzioni, labels=data_labels, pesi=nn_Struct.pesi, bias=nn_Struct.bias, learning_rate=lr)
        if(optim == "Adam"):
            pass




nn=NeuralNetArchitecture(input_data=data_features)


class NeuralNetwork():
    def __init__(self, epochs, learning_rate):
        self.epochs=epochs
        self.lr=learning_rate
        self.predizione=[]
        self.errori=[]
        self.epochi=[]
    
    def train(self):
        for epoch in range(self.epochs):

            #forward pass
            preds=nn.Forward()

            #calculate Loss
            loss=nn.calculateLoss(target=data_labels, predizione=preds, function="MSE")
            self.errori.append(loss)

            #backward pass
            nn.Backward(lr=0.001, optim="SGD")
            self.epochi.append(epoch)

            #prende le epochi e le errori per dopo visualizare il progresso 
            if epoch % 1 == 0:
                print(f"epoch: {epoch}, loss: {loss}")
            
            #previnire il overfitting interrompere prima
            if loss < 1.3:
                break
        
        #prende l'ultima predizione del modello
        self.predizione.append(preds)
        
    def predict(self, value):
        nn.input_data=value
        nn.Forward()
        print("predizione: nn.predizione")

nn_Model=NeuralNetwork(epochs=10, learning_rate=0.005)
nn_Model.train()
nn_Model.predict(value=[4e-6, 6e-6, 0.1])

print(f"\nlen training: {len(nn_Model.predizione)}\nlen di forza: {len(dataset.forza)}")

#plotando i dati, e avaluare i resultati
dataset.plotDataset(x=dataset.distanza, y=dataset.forza)
dataset.PlotModeloProgress(epochi=nn_Model.epochi, errori=nn_Model.errori)
dataset.comparezioneRisultato(predizioni=nn_Model.predizione, targets=dataset.forza)


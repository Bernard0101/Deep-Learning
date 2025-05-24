from src.Tools import functions as nn_func
import numpy as np  # type: ignore


class nn_Architettura:
    def __init__(self, nn_layers, init_pesi, inputs, features, targets, epoche, learning_rate):
        self.features=features
        self.targets=targets
        self.predizioni=[]
        self.pesi=[np.random.randn(nn_layers[0], features.shape[1])] + [np.random.randn(nn_layers[i], nn_layers[i-1]) for i in range(1, len(nn_layers))]
        self.bias=[np.random.randn(nn_layers[i])for i in range(len(nn_layers))] 
        self.inizializzazione=init_pesi
        self.nn_layers=nn_layers
        self.ativazioni=[]
        self.errori=[]
        self.epoche=epoche
        self.lr=learning_rate

    #implementa un layer qualsiasi della rete Neurale Multistrato
    def nn_ArcLayer(self, in_features, layer):
        pesi=self.pesi[layer]
        out_features=np.dot(in_features, pesi.T) + self.bias[layer]
        return out_features
    
    def initializzare_pesi(self):
        #implementa l'ativazzione Xavier
        if self.inizializzazione == "Xavier":
            Xavier_inizializzazione=np.sqrt(6 / (self.nn_layers[0] + self.nn_layers[-1]))
            self.pesi=[Xavier_inizializzazione * peso for peso in self.pesi]

        #implementa l'ativazzione He
        elif self.inizializzazione == "He":
            He_inizialiazzazione=np.sqrt(2 / (self.nn_layers[0]))
            self.pesi=[He_inizialiazzazione * peso for peso in self.pesi]
        else:
            raise ValueError(self.inizializzazione)

    #implementa la forward propagation d'accordo con il tipo di ativazione e initializzazione dei pesi
    def Forward(self):
        out_features=self.nn_ArcLayer(in_features=self.features, layer=0)
        out_features=nn_func.nn_functions.activation_leaky_ReLU(Z=out_features)
        self.ativazioni.append(out_features)

        out_features=self.nn_ArcLayer(in_features=out_features, layer=1)
        out_features=nn_func.nn_functions.activation_leaky_ReLU(Z=out_features)
        self.ativazioni.append(out_features)

        out_features=self.nn_ArcLayer(in_features=out_features, layer=2)
        out_features=nn_func.nn_functions.activation_leaky_ReLU(Z=out_features)
        self.ativazioni.append(out_features)

        out_features=self.nn_ArcLayer(in_features=out_features, layer=3)
        self.ativazioni.append(out_features)
        self.predizioni=out_features
        return out_features
    
    #implementa un modulo per calcolare lo sbaglio del modello basato in una metrica di avaluazione pre-scelta
    def Perdita(self):
        perdita=nn_func.nn_functions.Loss_MSE(y_pred=self.predizioni, y_label=self.targets)
        return perdita
    
    #implementa il modulo di Backpropagazione dove si addestrano i pesi della rete basatto in un'otimizzatore pre-scelto
    def Backward(self, funzione_ativazione):
        nn_func.nn_optimizers.optimizer_SGD(layers=self.nn_layers, ativazzioni=self.ativazioni,
                                            labels=self.targets, pesi=self.pesi, bias=self.bias, 
                                            lr=self.lr, ativazione=funzione_ativazione)
        
    def reset_parametri(self):
        self.predizioni=[]
        print(self.predizioni)
        self.errori=[]
        self.pesi=[np.random.randn(self.nn_layers[0], self.features.shape[1])] + [np.random.randn(self.nn_layers[i], self.nn_layers[i-1]) for i in range(1, len(self.nn_layers))]
        self.bias=[np.random.randn(self.nn_layers[i])for i in range(len(self.nn_layers))] 
        
    
    #loop di addestramento della rete d'accordo con la quantita di epoche
    def Allenare(self):
        self.reset_parametri()
        self.initializzare_pesi()
        for epoch in range(self.epoche):
            self.Forward()
            loss=self.Perdita()
            self.Backward(funzione_ativazione="leaky_ReLU")
            if epoch % 5 == 0:
                print(f"epoca: {epoch}| perdita: {loss}")
                self.errori.append(loss)
            
    

    






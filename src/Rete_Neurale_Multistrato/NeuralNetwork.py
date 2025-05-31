from src.Tools import functions as nn_func
import numpy as np  # type: ignore


class nn_Architettura:
    def __init__(self, nn_layers, init_pesi, features, targets, epoche, learning_rate, ottimizzattore, funzione_perdita):
        self.features=features
        self.targets=targets
        self.pesi=[np.random.randn(nn_layers[0], features.shape[1])] + [np.random.randn(nn_layers[i], nn_layers[i-1]) for i in range(1, len(nn_layers))]
        self.bias=[np.random.randn(nn_layers[i])for i in range(len(nn_layers))] 
        self.inizializzazione=init_pesi
        self.nn_layers=nn_layers
        self.ativazioni=[]
        self.predizioni=[]
        self.errori=[]
        self.epoche=epoche
        self.lr=learning_rate
        self.optim=ottimizzattore
        self.loss_fn=funzione_perdita

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
    def Forward(self, features):
        for layer in range(len(self.nn_layers)):
            out_features=self.nn_ArcLayer(in_features=features, layer=0)
            out_features=nn_func.nn_functions.activation_tanh(Z=out_features)
            self.ativazioni.append(out_features)
        return out_features
    
    #implementa un modulo per calcolare lo sbaglio del modello basato in una metrica di avaluazione pre-scelta
    def Perdita(self, predizioni, loss_fn):
        if loss_fn == "MSE":
            perdita_MSE=nn_func.nn_functions.Loss_MSE(y_pred=predizioni, y_label=self.targets)
            return perdita_MSE
        elif loss_fn == "MAE":
            perdita_MAE=nn_func.nn_functions.Loss_MAE(y_pred=predizioni, y_label=self.targets)
            return perdita_MAE
        elif loss_fn == "BCE":
            perdita_BCE=nn_func.nn_functions.Loss_BCE(y_pred=predizioni, y_label=self.targets)
            return perdita_BCE
    
    #implementa il modulo di Backpropagazione dove si addestrano i pesi della rete basatto in un'otimizzatore pre-scelto
    def Backward(self, optim):
        if optim == "SGD":
            nn_func.nn_optimizers.optimizer_SGD(layers=self.nn_layers, attivazzioni=self.ativazioni,
                                                targets=self.targets, pesi=self.pesi, bias=self.bias, 
                                                lr=self.lr)
        elif optim == "Adagrad":
            nn_func.nn_optimizers.optimizer_Adagrad(layers=self.nn_layers, attivazioni=self.ativazioni, 
                                                    targets=self.targets, pesi=self.pesi, bias=self.bias,
                                                    lr=self.lr)
        
    def reset_parametri(self):
        self.pesi=[np.random.randn(self.nn_layers[0], self.features.shape[1])] + [np.random.randn(self.nn_layers[i], self.nn_layers[i-1]) for i in range(1, len(self.nn_layers))]
        self.bias=[np.random.randn(self.nn_layers[i])for i in range(len(self.nn_layers))] 
        self.errori=[]
        
    
    #loop di addestramento della rete d'accordo con la quantita di epoche
    def Allenare(self):
        self.reset_parametri()
        self.initializzare_pesi()
        for epoch in range(self.epoche):
            #itera su features e targets fold, un'esempio alla volta
            preds=self.Forward(features=self.features)
            loss=self.Perdita(predizioni=preds, loss_fn=self.loss_fn)
            self.Backward(optim=self.optim)
            self.errori.append(loss)
            if epoch % 5 == 0:
                print(f"epoca: {epoch}| perdita: {loss}")


    def predict(self, features):
        predizione=self.Forward(features=features)
        return predizione


            
    

    






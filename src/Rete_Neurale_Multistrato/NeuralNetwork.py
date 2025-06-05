from src.Tools import functions as nn_func
import numpy as np  # type: ignore


class nn_Architettura:
    def __init__(self, nn_layers:list, init_pesi:str, features:np.ndarray, targets:np.ndarray, epoche:int, learning_rate:float, ottimizzattore:str, funzione_perdita:str, attivazione:str):
        self.features=features
        self.targets=targets
        self.pesi=[np.random.randn(nn_layers[0], features.shape[1])] + [np.random.randn(nn_layers[i], nn_layers[i-1]) for i in range(1, len(nn_layers))]
        self.bias=[np.random.randn(i) for i in nn_layers]
        self.inizializzazione=init_pesi
        self.nn_layers=nn_layers
        self.ativazioni=[]
        self.somme_pesate=[]
        self.predizioni=[]
        self.errori=[]
        self.epoche=epoche
        self.lr=learning_rate
        self.optim=ottimizzattore
        self.loss_fn=funzione_perdita
        self.activation_fn=attivazione

    #implementa un layer qualsiasi della rete Neurale Multistrato
    def nn_ArcLayer(self, in_features:np.ndarray, layer:int):
        pesi=self.pesi[layer]
        #print(f"pesi shape: {pesi.shape} features shape: {in_features.shape}")
        out_features=np.dot(in_features, pesi.T) + self.bias[layer]
        return out_features
    
    def initializzare_pesi(self, init_pesi:str):
        #implementa l'ativazzione Xavier
        if init_pesi == "Xavier":
            Xavier_inizializzazione=np.sqrt(6 / (self.nn_layers[0] + self.nn_layers[-1]))
            self.pesi=[Xavier_inizializzazione * peso for peso in self.pesi]

        #implementa l'ativazzione He
        elif init_pesi == "He":
            He_inizialiazzazione=np.sqrt(2 / (self.nn_layers[0]))
            self.pesi=[He_inizialiazzazione * peso for peso in self.pesi]
        else:
            raise ValueError(f"funzione di inizializzazione dei pesi: {init_pesi} non supportata")



    #implementa la forward propagation d'accordo con il tipo di ativazione e initializzazione dei pesi
    def Forward(self, features:np.ndarray):
        for layer in range(len(self.nn_layers)):
            #print(f"strato: {layer}")
            if layer == 0:
                Z=self.nn_ArcLayer(in_features=features, layer=layer)
                self.somme_pesate.append(Z)
                #print(f"somma pesata strato: {layer} -> {Z.shape}")
                out_features=nn_func.nn_functions.activation(nn_func.nn_functions, type=self.activation_fn, Z=Z, derivata=0)
                self.ativazioni.append(out_features)
                #print(f"attivazioni strato: {layer} -> {out_features.shape} ")
            else: 
                Z=self.nn_ArcLayer(in_features=out_features, layer=layer)
                self.somme_pesate.append(Z)
                #print(f"somma pesata strato: {layer} -> {Z.shape}")
                out_features=nn_func.nn_functions.activation(nn_func.nn_functions, type=self.activation_fn, Z=Z, derivata=0)
                self.ativazioni.append(out_features)
                #print(f"attivazioni starto: {layer} -> {out_features.shape} ")
        return out_features
    
    #implementa un modulo per calcolare lo sbaglio del modello basato in una metrica di avaluazione pre-scelta
    def Perdita(self, predizioni:np.ndarray, loss_fn:str):
        if loss_fn == "MSE":
            perdita_MSE=nn_func.nn_functions.Loss_MSE(y_pred=predizioni, y_label=self.targets)
            return perdita_MSE
        elif loss_fn == "MAE":
            perdita_MAE=nn_func.nn_functions.Loss_MAE(y_pred=predizioni, y_label=self.targets)
            return perdita_MAE
        elif loss_fn == "BCE":
            perdita_BCE=nn_func.nn_functions.Loss_BCE(y_pred=predizioni, y_label=self.targets)
            return perdita_BCE
        else:
            raise ValueError(f"funzione di costo {loss_fn} non supportata")

    
    #implementa il modulo di Backpropagazione dove si addestrano i pesi della rete basatto in un'otimizzatore pre-scelto
    def Backward(self, optim:str):
        if optim == "SGD":
            nn_func.nn_optimizers.optimizer_SGD(layers=self.nn_layers, attivazzioni=self.ativazioni, somme_pesate=self.somme_pesate,
                                                targets=self.targets, pesi=self.pesi, bias=self.bias, 
                                                lr=self.lr, activation_fn=self.activation_fn, loss_fn=self.loss_fn)
        elif optim == "Adagrad":
            nn_func.nn_optimizers.optimizer_Adagrad(layers=self.nn_layers, attivazioni=self.ativazioni, some_pesate=self.somme_pesate,
                                                targets=self.targets, pesi=self.pesi, bias=self.bias,
                                                lr=self.lr)
        else:
            raise ValueError(f"ottimizzattore {optim} non supportato")
        
    def reset_parametri(self):
        self.pesi=[np.random.randn(self.nn_layers[0], self.features.shape[1])] + [np.random.randn(self.nn_layers[i], self.nn_layers[i-1]) for i in range(1, len(self.nn_layers))]
        self.bias=[np.random.randn(self.nn_layers[i])for i in range(len(self.nn_layers))] 
        self.errori=[]
        
    
    #loop di addestramento della rete d'accordo con la quantita di epoche
    def Allenare(self):
        self.reset_parametri()
        self.initializzare_pesi(init_pesi=self.inizializzazione)
        for epoch in range(self.epoche):
            #itera su features e targets fold, un'esempio alla volta
            preds=self.Forward(features=self.features)
            loss=self.Perdita(predizioni=preds, loss_fn=self.loss_fn)
            self.Backward(optim=self.optim)
            self.errori.append(loss)
            if epoch % 5 == 0:
                print(f"epoca: {epoch}| perdita: {loss}")


    def predict(self, features:np.ndarray):
        predizione=self.Forward(features=features)
        return predizione


            
    

    






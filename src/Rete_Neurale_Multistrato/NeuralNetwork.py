from src.Tools import functions as nn_func
import numpy as np  # type: ignore


class nn_Architettura:
    """
    Classe che implementa l'architettura di una rete neurale multistrato in puro python

    params:
        self.features (np.ndarray): i valori di input per la rete
        self.targets (np.ndarray): i valori a esseri predetti
        self.pesi (np.ndarray): i pesi per ogni strato della rete, che possono essere o vettori, o matrici o tensori
        self.bias (np.ndarray): i bias per ogni strato della rete, un vettore corrispondente al numero di neuroni nello strato
        self.inizializzazione (str): un iperparametro che indica come inizializzare i pesi
        self.nn_layers (list): una lista di int indicando la quantita di nodi in ogni strato
        self.attivazioni (list): una lista di float che prende ogni attvazione della rete
        self.somme_pesate (list): una lista di float che prende ogni somma_pesata senza attivazione della rete
        self.errori (list): una lista per mantenere tracci dell'apprendimento della rete
        self.epoche (int): la quantita di volte che si allenera il modello
        self.lr (float): la tassa di apprndimento del modello
        self.optim (str): un'iperparametro che sceglie il tipo di ottimizzattore che verra utilizzato 
        self.loss_fn (str): un'iperparametro che sceglie il tipo funzione di perdita verra utilizzata
        self.activation_fn (str): un'iperparametro che sceglie la funzione di attivazione a essere utilzzata
    """
    def __init__(self, nn_layers:list, init_pesi:str, features:np.ndarray, targets:np.ndarray, epoche:int, learning_rate:float, ottimizzattore:str, funzione_perdita:str, attivazione:str):
        self.features=features
        self.targets=targets
        self.pesi=[np.random.randn(nn_layers[0], features.shape[1])] + [np.random.randn(nn_layers[i], nn_layers[i-1]) for i in range(1, len(nn_layers))]
        self.bias=[np.random.randn(i) for i in nn_layers]
        self.inizializzazione=init_pesi
        self.nn_layers=nn_layers
        self.ativazioni=[]
        self.somme_pesate=[]
        self.errori=[]
        self.epoche=epoche
        self.lr=learning_rate
        self.optim=ottimizzattore
        self.loss_fn=funzione_perdita
        self.activation_fn=attivazione

    #implementa un layer qualsiasi della rete Neurale Multistrato
    def nn_ArcLayer(self, in_features:np.ndarray, layer:int):
        """
        l'algoritmo di somma pesata tra input e pesi nel rispettivo strato
        di iterazione

        params: 
            in_features (np.ndarray): i dati di input
            layer (int): lo strato corrente

        returns:
            out_features (np.ndarray): la somma pesata tra i pesi e inputs
        """
        pesi=self.pesi[layer]
        #print(f"pesi shape: {pesi.shape} features shape: {in_features.shape}")
        out_features=np.dot(in_features, pesi.T) + self.bias[layer]
        return out_features
    
    def initializzare_pesi(self, init_pesi:str):
        """
        algortimi di Strategia per inizializzare i pesi come He e Xavier

        params: 
            init_pesi (str):il tipo di attivazione ad essere utilizzata

        returns:
            None
        """
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
        """
        Algortimo di forward pass di una rete neurale multistrato, dove per ogni strato
        si compie l'algoritmo di somma pesata e si finisce con un'attivazione prescelta dall'utente

        params:
            features (np.ndarray): i dati di input per la rete
        
        returns:
            out_features (np.ndarray): il risultato della predizione del modello
        """
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
    def Perdita(self, predizioni:np.ndarray):
        """
        Comparazione dei valori della predizione con i valori target

        params: 
            predizione (np.ndarray): le predizione risultanti dall'ultimo strato della rete
        
        returns: 
            None
        """
        return nn_func.nn_functions.Loss(nn_func.nn_functions, y_pred=predizioni, y_target=self.targets,
                                  type=self.loss_fn, derivata=False)        
        

    
    #implementa il modulo di Backpropagazione dove si addestrano i pesi della rete basatto in un'otimizzatore pre-scelto
    def Backward(self, optim:str):
        """
        L'algoritmo di retropropagazione o backpropagation, con il dovuto ottimizattore scelto dall'utente

        params: 
            optim (str): l'ottimizzattore scelto dall'utente
        
        returns: 
            None
        """
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
        """
        resetta i parametri della rete cioe i pesi i bias e gli errori

        params:
            None

        returns: 
            None
        """
        self.pesi=[np.random.randn(self.nn_layers[0], self.features.shape[1])] + [np.random.randn(self.nn_layers[i], self.nn_layers[i-1]) for i in range(1, len(self.nn_layers))]
        self.bias=[np.random.randn(self.nn_layers[i])for i in range(len(self.nn_layers))] 
        self.errori=[]
        
    
    #loop di addestramento della rete d'accordo con la quantita di epoche
    def Allenare(self):
        """
        Utilizza tutti i moduli della rete per allenarla seguendo la logica di elaborazione di predizione e di allenamento 
        forward-pass > calcolo di perdita > Backward

        params: 
            None
        
        returns:
            None
        """
        self.reset_parametri()
        self.initializzare_pesi(init_pesi=self.inizializzazione)
        for epoch in range(self.epoche):
            #itera su features e targets fold, un'esempio alla volta
            preds=self.Forward(features=self.features)
            #print(f"preds: {preds}")
            loss=self.Perdita(predizioni=preds)
            #print(f"loss: {loss}")
            self.Backward(optim=self.optim)
            self.errori.append(loss)
            if epoch % 5 == 0:
                print(f"epoca: {epoch}| perdita: {loss}")


    def predict(self, features:np.ndarray):
        predizione=self.Forward(features=features)
        return predizione


            
    

    






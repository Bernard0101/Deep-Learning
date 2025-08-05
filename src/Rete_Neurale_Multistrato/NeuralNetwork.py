from src.Rete_Neurale_Multistrato import Autodifferenziattore
from src.Tools import functions as nn_functions

import numpy as np  # type: ignore


class Architettura:
    def __init__(self, nn_layers, init_pesi, X_train, y_train, epochs, learning_rate, ottimizzattore, funzione_perdita, attivazione, X_test=None, y_test=None):
        self.pesi=[np.random.randn(X_train.shape[1], nn_layers[0])] + [np.random.randn(nn_layers[i-1], nn_layers[i]) for i in range(1, len(nn_layers))]
        self.bias=[np.random.randn(1, i) for i in nn_layers]
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        self.inizializzazione=init_pesi
        self.nn_layers=nn_layers
        self.migliore_modello=None
        self.train_errori=[]
        self.test_errori=[]
        self.epoche=[]
        self.epochs=epochs
        self.lr=learning_rate
        self.optim=ottimizzattore
        self.loss_fn=funzione_perdita
        self.activation_fn=attivazione
        self.autodiff=Autodifferenziattore.Autodiff(nn_strati=nn_layers, pesi=self.pesi, bias=self.bias, activation_fn=attivazione, loss_fn=funzione_perdita)
        self.SommaPesata=nn_functions.SommaPesata(autodiff=self.autodiff, pesi=self.pesi, bias=self.bias)
        self.attivazione=nn_functions.attivazione(autodiff=self.autodiff, type=attivazione)
        self.perdita=nn_functions.Perdita(autodiff=self.autodiff, type=funzione_perdita)
        self.optimizer=nn_functions.optimizers(autodiff=self.autodiff, alg_optim=self.optim, pesi=self.pesi, bias=self.bias)

    #implementa un layer qualsiasi della rete Neurale Multistrato 
    def initializzare_pesi(self, init_pesi:str):
        if init_pesi == "Xavier":
            for i in range(len(self.pesi)):
                fan_in=self.pesi[i].shape[1]
                fan_out=self.pesi[i].shape[0]
                limite=np.sqrt(6 / (fan_in + fan_out))
                self.pesi[i]=np.random.uniform(-limite, limite, size=self.pesi[i].shape)

        elif init_pesi == "He":
            for i in range(len(self.pesi)):
                fan_in=self.pesi[i].shape[1]
                self.pesi[i]=np.random.randn(*self.pesi[i].shape) * np.sqrt(2. / fan_in)
        else:
            raise ValueError(f"funzione di inizializzazione dei pesi: {init_pesi} non supportata")


    def show_parametri(self, strato=0):
        if strato > (len(self.nn_layers)-1):
            return None
        else:
            print(f"strato: {strato}")
            print(f"pesi: {self.pesi[strato].shape}")
            print(f"bias: {self.bias[strato].shape}")
            self.show_parametri(strato+1)


    #implementa la forward propagation d'accordo con il tipo di ativazione e initializzazione dei pesi
    def Forward(self, inputs):
        predizione=None
        for layer in range(len(self.nn_layers)):
            #print(f"strato: {layer}")
            if layer == 0:
                Z=self.SommaPesata.func(inputs=inputs, strato=layer, derivata=False)
                out_features=self.attivazione.func(inputs=Z, strato=layer, derivata=False)

            elif layer > 0 and layer != len(self.nn_layers)-1: 
                Z=self.SommaPesata.func(inputs=out_features, strato=layer, derivata=False)
                out_features=self.attivazione.func(inputs=Z, strato=layer, derivata=False)
            else:
                Z=self.SommaPesata.func(inputs=out_features, strato=layer, derivata=False)
                predizione=self.attivazione.func(inputs=Z, strato=layer, derivata=False)

        return predizione
    
    #implementa un modulo per calcolare lo sbaglio del modello basato in una metrica di avaluazione pre-scelta
    def Perdita(self, predizioni:np.ndarray, targets):
        loss=self.perdita.func(y_pred=predizioni, y_target=targets, derivata=False)
        self.autodiff.memorizzare(strato=(len(self.nn_layers)-1), inputs=[predizioni, targets], outputs=loss, operazione="Perdita")
        return loss
        
    
    #implementa il modulo di Backpropagazione dove si addestrano i pesi della rete basatto in un'otimizzatore pre-scelto
    def Backward(self, features, targets, predizioni):
        self.autodiff.retropropagazione(predizioni=predizioni, targets=targets, features=features)
        self.optimizer.algorithm(pesi=self.pesi, bias=self.bias, lr=self.lr)
      

    def reset_parametri(self):
        self.pesi=[np.random.randn(self.X_train.shape[1], self.nn_layers[0])] + [np.random.randn(self.nn_layers[i-1], self.nn_layers[i]) for i in range(1, len(self.nn_layers))]
        self.bias=[np.random.randn(1, i) for i in self.nn_layers]
        self.SommaPesata.pesi=self.pesi
        self.SommaPesata.bias=self.bias
        self.autodiff.passaggi=[]
        self.train_errori=[]
        self.test_errori=[]
        self.epoche=[]

    def salvare_modello(self, epoca, patience):
        if epoca > patience:
           self.migliore_modello=self


    def early_stop(self, epoca, patience, min_delta=-5e-2):
        if epoca < patience:
            return False

        migliore_alleno=min(self.train_errori[(epoca-patience):epoca])
        current=self.train_errori[epoca]
        differenza=migliore_alleno - current
        if differenza < min_delta or np.isnan(current):
            return True
        return False        


    def allenare(self, inputs): 
        y_pred=self.Forward(inputs=inputs)
        loss=self.Perdita(predizioni=y_pred, targets=self.y_train)
        self.Backward(features=self.X_train, targets=self.y_train, predizioni=y_pred)
        self.train_errori.append(loss)
        return loss

    def valutare(self, inputs):
        y_pred=self.Forward(inputs=inputs)
        loss=self.Perdita(predizioni=y_pred, targets=self.y_test)
        self.autodiff.passaggi=[]
        self.test_errori.append(loss)
        return y_pred
    
    #loop di addestramento della rete d'accordo con la quantita di epoche
    def Allenare(self):
        self.reset_parametri()
        self.initializzare_pesi(init_pesi=self.inizializzazione)
        for epoch in range(self.epochs):
            self.allenare(inputs=self.X_train)
            self.valutare(inputs=self.X_test)
            self.epoche.append(epoch)
            print(f"epoca: {epoch}| loss alleno: {self.train_errori[epoch]} | loss valutazione {self.test_errori[epoch]}")
            if(self.early_stop(epoca=epoch, patience=30)):
                self.salvare_modello(epoca=epoch, patience=30)  
                return 
        self.salvare_modello(epoca=epoch, patience=30)
        
        

    
    

            

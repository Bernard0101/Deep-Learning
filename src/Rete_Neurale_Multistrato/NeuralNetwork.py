from src.Rete_Neurale_Multistrato import Autodifferenziattore
from src.Tools import functions as nn_functions
import numpy as np  # type: ignore


class Architettura:
    def __init__(self, nn_layers:list, init_pesi:str, features:np.ndarray, targets:np.ndarray, epochs:int, learning_rate:float, ottimizzattore:str, funzione_perdita:str, attivazione:str):
        self.pesi=[np.random.randn(features.shape[1], nn_layers[0])] + [np.random.randn(nn_layers[i-1], nn_layers[i]) for i in range(1, len(nn_layers))]
        self.bias=[np.random.randn(1, i) for i in nn_layers]
        self.features=features
        self.targets=targets
        self.inizializzazione=init_pesi
        self.nn_layers=nn_layers
        self.preds=None
        self.errori=[]
        self.epoche=[]
        self.epochs=epochs
        self.lr=learning_rate
        self.optim=ottimizzattore
        self.loss_fn=funzione_perdita
        self.activation_fn=attivazione
        self.autodiff=Autodifferenziattore.Autodiff(nn_strati=nn_layers, pesi=self.pesi, bias=self.bias, activation_fn=attivazione, loss_fn=funzione_perdita, batch=self.features.shape[0])
        self.SommaPesata=nn_functions.SommaPesata(pesi=self.pesi, bias=self.bias)
        self.attivazione=nn_functions.attivazione(type=attivazione)
        self.perdita=nn_functions.Perdita(type=funzione_perdita)
        self.optimizer=nn_functions.optimizers(alg_optim=self.optim, pesi=self.pesi, bias=self.bias, grad_pesi=self.autodiff.gradiente_pesi, grad_bias=self.autodiff.gradiente_bias)

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
                self.autodiff.memorizzare(strato=layer, inputs=inputs, outputs=Z, operazione="somma_pesata")
                out_features=self.attivazione.func(inputs=Z, type=self.activation_fn, derivata=False)
                self.autodiff.memorizzare(strato=layer, inputs=Z, outputs=out_features, operazione="attivazione")

            elif layer > 0 and layer != len(self.nn_layers)-1: 
                Z=self.SommaPesata.func(inputs=out_features, strato=layer, derivata=False)
                self.autodiff.memorizzare(strato=layer, inputs=out_features, outputs=Z, operazione="somma_pesata")
                out_features=self.attivazione.func(inputs=Z, type=self.activation_fn, derivata=False)
                self.autodiff.memorizzare(strato=layer, inputs=Z, outputs=out_features, operazione="attivazione")
            else:
                Z=self.SommaPesata.func(inputs=out_features, strato=layer, derivata=False)
                self.autodiff.memorizzare(strato=layer, inputs=out_features, outputs=Z, operazione="somma_pesata")
                predizione=self.attivazione.func(inputs=Z, type=self.activation_fn, derivata=False)
                self.autodiff.memorizzare(strato=layer, inputs=Z, outputs=predizione, operazione="attivazione")

        return predizione
    
    #implementa un modulo per calcolare lo sbaglio del modello basato in una metrica di avaluazione pre-scelta
    def Perdita(self, predizioni:np.ndarray, targets):
        loss=self.perdita.func(y_pred=predizioni, y_target=targets, type=self.loss_fn, derivata=False)
        self.autodiff.memorizzare(strato=(len(self.nn_layers)-1), inputs=[predizioni, targets], outputs=loss, operazione="Perdita")
        return loss
        
    
    #implementa il modulo di Backpropagazione dove si addestrano i pesi della rete basatto in un'otimizzatore pre-scelto
    def Backward(self, features, targets, predizioni):
        self.autodiff.retropropagazione(predizioni=predizioni, targets=targets, features=features)
        self.optimizer.func(pesi=self.pesi, bias=self.bias, lr=self.lr, type=self.optim)
      

    def reset_parametri(self):
        self.pesi=[np.random.randn(self.features.shape[1], self.nn_layers[0])] + [np.random.randn(self.nn_layers[i-1], self.nn_layers[i]) for i in range(1, len(self.nn_layers))]
        self.bias=[np.random.randn(1, i) for i in self.nn_layers]
        self.SommaPesata.pesi=self.pesi
        self.SommaPesata.bias=self.bias
        self.autodiff.passaggi=[]
        self.errori=[]


    def regolarizzazione(self, epoca, patience, min_delta=-1e-3):
        if epoca < patience:
            return False

        migliore_alleno=min(self.errori[epoca - patience:epoca])
        current=self.errori[epoca]
        if migliore_alleno - current < min_delta or np.isnan(current):
            return True
        return False        
        

    def shuffle_data(self):
        indices=np.arange(len(self.features))
        np.random.shuffle(indices)
        features, targets=self.features[indices], self.targets[indices]
        return features, targets

                  
    #loop di addestramento della rete d'accordo con la quantita di epoche
    def Allenare(self):
        self.reset_parametri()
        self.initializzare_pesi(init_pesi=self.inizializzazione)
        features, targets=self.shuffle_data()
        for epoch in range(self.epochs):
            y_preds=self.Forward(inputs=features)
            loss=self.Perdita(targets=targets, predizioni=y_preds)
            self.Backward(features=features, targets=targets, predizioni=y_preds)
            self.errori.append(loss)
            print(f"epoca: {epoch}| errore: {loss}")
            if(self.regolarizzazione(epoca=epoch, patience=15)):
                break
        self.epoche.append(epoch)
            

    def predict(self, inputs):
        predizione=self.Forward(inputs=inputs)
        self.preds=predizione
        return predizione
    

            

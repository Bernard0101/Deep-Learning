import numpy as np

from src.Rete_Neurale_Multistrato import NeuralNetwork as nn
from src.Rete_Neurale_Multistrato import Autodifferenziattore
from src.Tools import functions as nn_functions


class Physics_informedNN(nn.Architettura):
    def __init__(self,  nn_layers:list, init_pesi:str, features:np.ndarray, targets:np.ndarray, epochs:int, learning_rate:float, ottimizzattore:str, funzione_perdita:str, attivazione:str):
        super.__init__()
        self.pesi=[np.random.randn(nn_layers[0], features.shape[1])] + [np.random.randn(nn_layers[i], nn_layers[i-1]) for i in range(1, len(nn_layers))]
        self.bias=[np.random.randn(i, 1) for i in nn_layers]
        self.features=features
        self.targets=targets
        self.inizializzazione=init_pesi
        self.nn_layers=nn_layers
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
        self.optimizer=nn_functions.optimizers(alg_optim=self.optim, grad_pesi=self.autodiff.gradiente_pesi, grad_bias=self.autodiff.gradiente_bias)


    def Perdita(self, predizioni, termine_fisico, delta=3e-3):
        loss_fisica=self.perdita.func(y_pred=None, y_target=termine_fisico, type=self.loss_fn, derivata=False)
        loss_dati=self.perdita.func(y_pred=predizioni, y_target=self.targets, type=self.loss_fn, derivata=False)
        perdita_complessiva=loss_dati + loss_fisica * delta
        return perdita_complessiva


    def Allenare(self):
        self.reset_parametri()
        self.initializzare_pesi(init_pesi=self.inizializzazione)
        for epoch in range(self.epochs):
            y_preds=self.Forward(inputs=self.features)
            loss=self.Perdita(predizioni=y_preds)
            self.Backward(predizioni=y_preds)
            self.errori.append(loss)
            print(f"epoca: {epoch}| perdita: {loss}")
            if(self.regolarizzazione(epoca=epoch, patience=15)):
                break
            self.epoche.append(epoch)

        

        
        
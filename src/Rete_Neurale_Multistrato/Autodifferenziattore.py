import numpy as np

from src.Tools import functions as nn_functions


class Autodiff:
    def __init__(self, nn_strati, activation_fn, loss_fn, pesi, bias):
        self.SommaPesata=nn_functions.SommaPesata(autodiff=self, pesi=pesi, bias=bias)
        self.attivazione=nn_functions.attivazione(autodiff=self, type=activation_fn)
        self.Perdita=nn_functions.Perdita(autodiff=self, type=loss_fn)
        self.pesi=pesi
        self.bias=bias
        self.passaggi=[]
        self.nn_strati=nn_strati
        self.strati=(len(nn_strati)-1)
        self.activation_fn=activation_fn
        self.loss_fn=loss_fn
        self.gradiente_pesi=[np.zeros_like(p)for p in pesi]
        self.gradiente_bias=[np.zeros_like(b)for b in bias]
        self.gradiente_output=None
        
    
    def memorizzare(self, strato, inputs, outputs, operazione):
        passaggio= {"strato":strato,
                    "operazione":operazione,
                    "inputs":inputs,
                    "outputs":outputs}
        self.passaggi.append(passaggio)
    

    def show_passaggi(self):
        for passaggio in self.passaggi:
            for key, value in passaggio.items():
                if key == "strato": 
                    print(f"strato: {value}")
                elif key == "operazione":
                    print(f"operazione: {value}")
                elif key == "inputs" and not isinstance(value, list):
                    print(f"inputs shape: {value.shape}")
                elif key == "outputs" and not isinstance(value, list):
                    print(f"outputs shape: {value.shape}")


    def retropropagazione(self, predizioni, features, targets):
        for passaggio_idx in reversed(range(len(self.passaggi))):
            strato=self.passaggi[passaggio_idx]["strato"]
            operazione=self.passaggi[passaggio_idx]["operazione"]

            if operazione == "Perdita":
                gradiente_loss=self.Perdita.func(y_pred=predizioni, y_target=targets, type=self.loss_fn, derivata=True)

            elif operazione == "attivazione":
                attivazioni_precedenti=self.passaggi[passaggio_idx-2]["inputs"]
                Z=self.passaggi[passaggio_idx]["inputs"] 
                gradiente_attivazione=self.attivazione.func(inputs=Z, strato=strato, type=self.activation_fn, derivata=True)

            elif operazione == "somma_pesata":
                if strato == self.strati:
                    #print(f"grad_loss: {gradiente_loss.shape} grad_attivazione: {gradiente_attivazione.shape}")
                    gradiente_delta=(gradiente_loss * gradiente_attivazione)
                    #print(f"gradiente delta: {gradiente_delta.shape}")
                    #print(f"attivazioni precedenti trasposte: {attivazioni_precedenti.T.shape} gradiente delta: {gradiente_delta.shape}")
                    #print(f"gradiente delta: {gradiente_delta.shape} gradiente bias: {(np.ones(gradiente_delta.shape[0]).shape)}")
                    self.gradiente_pesi[strato]=np.dot(attivazioni_precedenti.T, gradiente_delta)
                    self.gradiente_bias[strato]=np.sum(gradiente_delta, axis=0, keepdims=True) 
                    #print(f"gradiente pesi: {self.gradiente_pesi[strato].shape}")
                    #print(f"gradiente bias: {self.gradiente_bias[strato].shape}")

                elif strato < self.strati and strato != 0: 
                    #print(f"gradiente delta: {gradiente_delta.shape} pesi strato sucessivo: {self.pesi[strato+1].T.shape} gradiente attivazione: {gradiente_attivazione.shape}")
                    gradiente_delta=np.dot(gradiente_delta, self.pesi[strato+1].T) * gradiente_attivazione
                    #print(f"gradiente delta: {gradiente_delta.shape}")
                    #print(f"attivazioni precedenti trasposte: {attivazioni_precedenti.T.shape} gradiente delta: {gradiente_delta.shape}")
                    #print(f"gradiente delta: {gradiente_delta.shape} gradiente bias: {(np.ones(gradiente_delta.shape[1]).shape)}")
                    self.gradiente_pesi[strato]=np.dot(attivazioni_precedenti.T, gradiente_delta)
                    self.gradiente_bias[strato]=np.sum(gradiente_delta, axis=0, keepdims=True) 
                    #print(f"gradiente pesi: {self.gradiente_pesi[strato].shape}")
                    #print(f"gradiente bias: {self.gradiente_bias[strato].shape}")
                
                elif strato == 0:
                    #print(f"gradiente delta: {gradiente_delta.shape} pesi strato sucessivo: {self.pesi[strato+1].T.shape} gradiente attivazione: {gradiente_attivazione.shape}")
                    gradiente_delta=np.dot(gradiente_delta, self.pesi[strato+1].T) * gradiente_attivazione
                    #print(f"gradiente delta: {gradiente_delta.shape}")
                    #print(f"inputs: {features.T.shape} gradiente delta: {gradiente_delta.shape}")
                    #print(f"gradiente delta: {gradiente_delta.shape} gradiente bias: {(np.ones(gradiente_delta.shape[1]).shape)}")
                    self.gradiente_pesi[strato]=np.dot(features.T, gradiente_delta) 
                    self.gradiente_bias[strato]=np.sum(gradiente_delta, axis=0, keepdims=True) 
                    #print(f"gradiente pesi: {self.gradiente_pesi[strato].shape}")
                    #print(f"gradiente bias: {self.gradiente_bias[strato].shape}")
                    self.passaggi=[]
            else:
                raise ValueError(f"operazione {self.passaggi[passaggio_idx]["operazione"]} non supportata")
    


    def show_gradients(self, strato=0):
        if strato > self.strati:
            return None
        else:
            print(f"strato: {strato}")
            print(f"grad_pesi: {self.gradiente_pesi[strato].shape}")
            print(f"grad_bias: {self.gradiente_bias[strato].shape}")
            self.show_gradients(strato+1)
            
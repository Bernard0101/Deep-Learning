import numpy as np

from src.Tools import functions as nn_functions


class Autodiff:
    def __init__(self, strati, activation_fn, loss_fn, pesi, bias):
        self.SommaPesata=nn_functions.SommaPesata(pesi=pesi, bias=bias)
        self.attivazione=nn_functions.attivazione(type=activation_fn)
        self.Perdita=nn_functions.Perdita(type=loss_fn)
        self.pesi=pesi
        self.bias=bias
        self.gradiente_pesi=[np.ones_like(p)for p in pesi]
        self.gradiente_bias=[np.ones_like(b)for b in bias]
        self.passaggi=[]
        self.strati=strati
        self.activation_fn=activation_fn
        self.loss_fn=loss_fn
       


    def memorizzare(self, strato, inputs, outputs, operazione:str):
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
                    continue
                elif key == "operazione":
                    print(f"operazione: {value}")
                    continue
                elif key == "inputs" and not isinstance(value, list):
                    print(f"inputs shape: {value.shape}")
                    continue
                elif key == "outputs" and not isinstance(value, list):
                    print(f"outputs shape: {value.shape}")
                

    def retropropagazione(self, predizioni, targets):
        calculate_grad=len(self.passaggi)-3
        for passaggio_idx in reversed(range(len(self.passaggi))):
            strato=self.passaggi[passaggio_idx]["strato"]
            #print(f"strato: {strato}")
            #print(f"passaggio: {passaggio_idx}")


            if self.passaggi[passaggio_idx]["operazione"] == "Perdita":
                gradiente_loss=self.Perdita.func(y_pred=predizioni, y_target=targets, type=self.loss_fn, derivata=True)

            elif self.passaggi[passaggio_idx]["operazione"] == "attivazione":
                Z=self.passaggi[passaggio_idx]["inputs"] 
                attivazioni_precedenti=self.passaggi[passaggio_idx]["outputs"]
                gradiente_attivazione=self.attivazione.func(inputs=Z, type=self.activation_fn, derivata=True)

            if strato == self.strati and passaggio_idx == (len(self.passaggi)-3):
                #print(f"grad_loss: {gradiente_loss.shape} grad_attivazione: {gradiente_attivazione.shape}")
                gradiente_delta=(gradiente_loss * gradiente_attivazione)
                #print(f"gradiente delta: {gradiente_delta.shape}")
                self.gradiente_pesi[strato]=(gradiente_delta * attivazioni_precedenti)
                #print(f"gradiente pesi: {self.gradiente_pesi[strato].shape}")
                calculate_grad-=2

            elif strato < self.strati and passaggio_idx == calculate_grad: 
                #print(f"gradiente delta: {gradiente_delta.shape} pesi strato sucessore: {self.pesi[strato+1].shape} gradiente attivazione: {gradiente_attivazione.shape}")
                gradiente_delta=np.dot(gradiente_delta, self.pesi[strato+1]) * gradiente_attivazione
                #print(f"gradiente delta: {gradiente_delta.shape}")
                self.gradiente_pesi[strato]=(gradiente_delta * attivazioni_precedenti)
                #print(f"gradiente pesi: {self.gradiente_pesi[strato].shape}")
                calculate_grad-=2


    def show_gradients(self, strato):
        if strato > self.strati:
            return
        else:
            print(f"strato: {strato} grad_shape: {self.gradiente_pesi[strato].shape}")
            self.show_gradients(strato+1)
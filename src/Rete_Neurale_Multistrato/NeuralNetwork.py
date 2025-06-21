from src.Tools import functions as nn_functions
from src.Tools import PIML as fisica_func
import numpy as np  # type: ignore


class nn_Architettura:
    def __init__(self, nn_layers:list, init_pesi:str, features:np.ndarray, targets:np.ndarray, epochs:int, learning_rate:float, ottimizzattore:str, funzione_perdita:str, attivazione:str):
        self.pesi=[np.random.randn(nn_layers[0], features.shape[1])] + [np.random.randn(nn_layers[i], nn_layers[i-1]) for i in range(1, len(nn_layers))]
        self.bias=[np.random.randn(i, 1) for i in nn_layers]
        self.autodiff=Autodifferenziattore(strati=(len(nn_layers)-1), pesi=self.pesi, bias=self.bias, activation_fn=attivazione, loss_fn=funzione_perdita)
        self.features=features
        self.targets=targets
        self.inizializzazione=init_pesi
        self.nn_layers=nn_layers
        self.errori=[]
        self.epoche=[]
        self.SommaPesata=nn_functions.SommaPesata(pesi=self.pesi, bias=self.bias)
        self.attivazione=nn_functions.attivazione(type=attivazione)
        self.Perdita=nn_functions.Perdita(type=funzione_perdita)
        self.optimizer=nn_functions.optimizers(alg_optim=ottimizzattore, pesi=self.pesi, bias=self.bias, lr=learning_rate, grad_pesi=self.autodiff.gradiente_pesi, grad_bias=self.autodiff.gradiente_bias)
        self.epochs=epochs
        self.lr=learning_rate
        self.optim=ottimizzattore
        self.loss_fn=funzione_perdita
        self.activation_fn=attivazione

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


    #implementa la forward propagation d'accordo con il tipo di ativazione e initializzazione dei pesi
    def Forward(self, inputs):
        predizione=None
        for layer in range(len(self.nn_layers)):
            print(f"strato: {layer}")
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
    def perdita(self, predizioni:np.ndarray):
        loss=self.Perdita.func(y_pred=predizioni, y_target=self.targets, type=self.loss_fn, derivata=False)
        self.autodiff.memorizzare(strato=(len(self.nn_layers)-1), inputs=[predizioni, self.targets], outputs=loss, operazione="Perdita")
        return loss
        
    
    #implementa il modulo di Backpropagazione dove si addestrano i pesi della rete basatto in un'otimizzatore pre-scelto
    def Backward(self):
        self.optimizer.optimizer_SGD()
      

    def reset_parametri(self):
        self.pesi=[np.random.randn(self.nn_layers[0], self.features.shape[1])] + [np.random.randn(self.nn_layers[i], self.nn_layers[i-1]) for i in range(1, len(self.nn_layers))]
        self.bias=[np.random.randn(i) for i in self.nn_layers]
        self.SommaPesata.pesi=self.pesi
        self.SommaPesata.bias=self.bias
        self.autodiff.passaggi=[]
        self.errori=[]


    def regolarizzazione(self, epoca, patience, min_delta=-1e-4):
        if epoca < patience:
            return False

        migliore_alleno=min(self.errori[epoca - patience:epoca])
        current=self.errori[epoca]
        if migliore_alleno - current < min_delta:
            return True
        return False        
        
                  
    #loop di addestramento della rete d'accordo con la quantita di epoche
    def Allenare(self):
        self.reset_parametri()
        self.initializzare_pesi(init_pesi=self.inizializzazione)
        for epoch in range(self.epochs):
            preds=self.Forward(inputs=self.features)
            print(f"pred: {preds.shape}")
            loss=self.perdita(predizioni=preds)
            self.autodiff.show_passaggi()
            self.autodiff.retropropagazione(predizioni=preds, targets=self.targets)
            self.Backward()
            self.errori.append(loss)
            print(f"epoca: {epoch}| perdita: {loss}")
            if(self.regolarizzazione(epoca=epoch, patience=15)):
                break
        self.epoche.append(epoch)
            #if epoch % 5 == 0:
              #      for p in self.pesi:
               #         print(f"media dei pesi: {np.mean(p)}")

    def predict(self, features:np.ndarray):
        predizione=self.Forward(features=features)
        return predizione


            
    

class Autodifferenziattore:
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
            print(f"strato: {strato}")
            print(f"passaggio: {passaggio_idx}")


            if self.passaggi[passaggio_idx]["operazione"] == "Perdita":
                gradiente_loss=self.Perdita.func(y_pred=predizioni, y_target=targets, type=self.loss_fn, derivata=True)

            elif self.passaggi[passaggio_idx]["operazione"] == "attivazione":
                Z=self.passaggi[passaggio_idx]["inputs"] 
                attivazioni_precedenti=self.passaggi[passaggio_idx]["outputs"]
                gradiente_attivazione=self.attivazione.func(inputs=Z, type=self.activation_fn, derivata=True)

            if strato == self.strati and passaggio_idx == (len(self.passaggi)-3):
                print(f"grad_loss: {gradiente_loss.shape} grad_attivazione: {gradiente_attivazione.shape}")
                gradiente_delta=gradiente_loss * gradiente_attivazione
                print(f"gradiente delta: {gradiente_delta.shape}")
                self.gradiente_pesi[strato]=np.dot(gradiente_delta, attivazioni_precedenti.T)
                self.gradiente_bias[strato]=None
                calculate_grad-=2

            elif strato < self.strati and passaggio_idx == calculate_grad and passaggio_idx != 0: 
                print(f"gradiente delta: {gradiente_delta.shape} pesi strato sucessore: {self.pesi[strato+1].shape} gradiente attivazione: {gradiente_attivazione.shape}")
                gradiente_delta=np.dot(gradiente_delta, self.pesi[strato+1]) * gradiente_attivazione
                print(f"gradiente delta: {gradiente_delta.shape}")
                self.gradiente_pesi[strato]=np.dot(gradiente_delta, attivazioni_precedenti.T)
                calculate_grad-=2
                
                
           

                



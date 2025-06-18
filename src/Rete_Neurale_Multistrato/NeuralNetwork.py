from src.Tools import functions as nn_func
from src.Tools import PIML as fisica_func
import numpy as np  # type: ignore


class nn_Architettura:
    def __init__(self, nn_layers:list, init_pesi:str, features:np.ndarray, targets:np.ndarray, epochs:int, learning_rate:float, ottimizzattore:str, funzione_perdita:str, attivazione:str):
        self.features=features
        self.targets=targets
        self.pesi=[np.random.randn(nn_layers[0], features.shape[1])] + [np.random.randn(nn_layers[i], nn_layers[i-1]) for i in range(1, len(nn_layers))]
        self.bias=[np.random.randn(i) for i in nn_layers]
        self.inizializzazione=init_pesi
        self.nn_layers=nn_layers
        self.ativazioni=[]
        self.somme_pesate=[]
        self.errori=[]
        self.epoche=[]
        self.epochs=epochs
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
        if init_pesi == "Xavier":
            for i in range(len(self.pesi)):
                fan_in = self.pesi[i].shape[1]
                fan_out = self.pesi[i].shape[0]
                limite = np.sqrt(6 / (fan_in + fan_out))
                self.pesi[i] = np.random.uniform(-limite, limite, size=self.pesi[i].shape)

        elif init_pesi == "He":
            for i in range(len(self.pesi)):
                fan_in = self.pesi[i].shape[1]
                self.pesi[i] = np.random.randn(*self.pesi[i].shape) * np.sqrt(2. / fan_in)
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
                out_features=nn_func.nn_functions.activation(nn_func.nn_functions, type=self.activation_fn, Z=Z, derivata=False)
                self.ativazioni.append(out_features)
                #print(f"attivazioni strato: {layer} -> {out_features.shape} ")
            elif layer > 0 and layer < len(self.nn_layers): 
                Z=self.nn_ArcLayer(in_features=out_features, layer=layer)
                self.somme_pesate.append(Z)
                #print(f"somma pesata strato: {layer} -> {Z.shape}")
                out_features=nn_func.nn_functions.activation(nn_func.nn_functions, type=self.activation_fn, Z=Z, derivata=False)
                self.ativazioni.append(out_features)
                #print(f"attivazioni starto: {layer} -> {out_features.shape} ")
            else:
                Z=self.nn_ArcLayer(in_features=out_features, layer=layer)
                self.somme_pesate.append(out_features)
                #print(f"somma pesata strato: {layer} -> {Z.shape}")
        return out_features
    
    #implementa un modulo per calcolare lo sbaglio del modello basato in una metrica di avaluazione pre-scelta
    def Perdita(self, predizioni:np.ndarray):
        loss_fisica=fisica_func.Fisica.MSE_leggeCoulomb(fisica_func.Fisica, y_pred=predizioni, features=self.features)

        loss_dati=nn_func.nn_functions.Loss(nn_func.nn_functions, y_pred=predizioni, y_target=self.targets,
                                  type=self.loss_fn, derivata=False)
        importanza=1e-8
        
        loss_complessiva=loss_dati + loss_fisica * importanza
        return loss_complessiva
        
    
    #implementa il modulo di Backpropagazione dove si addestrano i pesi della rete basatto in un'otimizzatore pre-scelto
    def Backward(self, optim:str):

        if optim == "SGD":
            nn_func.nn_optimizers.optimizer_SGD(layers=self.nn_layers, attivazzioni=self.ativazioni, somme_pesate=self.somme_pesate,
                                                targets=self.targets, features=self.features, pesi=self.pesi, bias=self.bias, 
                                                lr=self.lr, activation_fn=self.activation_fn, loss_fn=self.loss_fn, legge_fisica=None)
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
            #itera su features e targets fold, un'esempio alla volta
            preds=self.Forward(features=self.features)
            #print(f"preds: {preds}")
            loss=self.Perdita(predizioni=preds)
            #print(f"loss: {loss}")
            self.Backward(optim=self.optim)
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
    def __init__(self, strati, ):
        self.passaggi=[]    

    def decodificare_operazione(self, operazione):
        if 


    def memorizzare(self, inputs, outputs, operazione):
        passaggio=[
            {"inputs" : inputs},
            {"outputs" : outputs},
            {"operazione" : operazione}
            ]
        self.passaggi.append(passaggio)









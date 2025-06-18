from src.Tools.PIML import Fisica
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

class Funzioni_SommaPesata:
    def __init__(self, X_inputs, W_pesi):
        self.operazione="SommaPesata"
        self.inputs=X_inputs
        self.pesi=W_pesi
        self.output=None

    def func(self, derivata):
        if not derivata:
            return Funzioni_SommaPesata.prodotto_matriciale(X=self.inputs, W=self.pesi)
        else:    
            return Funzioni_SommaPesata.derivata_prodotto_matriciale(W=self.pesi)

    def prodotto_matriciale(X, W):
        return np.dot(X, W)
            
    
    def derivata_prodotto_matriciale(W):
        return W.T


class Funzioni_attivazione:
    def __init__(self, Z):
        self.operazione="attivazione"
        self.input_Z=Z
        self.output=None

    def func(self, type:str, derivata:bool):
        if type == "ReLU":
            if (not derivata):
                return self.activation_ReLU(Z=self.input_Z)
            else:
                return self.activation_ReLU_derivative(Z=self.input_Z)
        elif type == "leaky_ReLU":
            if(not derivata):
                return self.activation_leaky_ReLU(Z=self.input_Z)
            else:
                return self.activation_leaky_ReLU_derivative(Z=self.input_Z)
        elif type == "Sigmoid":
            if (not derivata):
                return self.activation_Sigmoid(Z=self.input_Z)
            else: 
                return self.activation_Sigomid_derivative(Z=self.input_Z)
        elif type == "Tanh":
            if (not derivata):
                return self.activation_tanh(Z=self.input_Z)
            else: 
                return self.activation_tanh_derivative(Z=self.input_Z)
        else:
            raise ValueError(f"la funzione {type} non e supportata")

    #ReLU function ativazione
    def activation_ReLU(self, Z):
        result=np.maxpassimum(0, Z)
        self.output=result
        return result

    def activation_ReLU_derivative(self, Z):
        self.output=np.where(Z > 0, 1, 0)
        return self.output

    #Leaky ReLU variant ativazione
    def activation_leaky_ReLU(self, Z, alpha=0.03):
        self.output=np.where(Z >= 0, Z, alpha * Z)
        return self.output

    def activation_leaky_ReLU_derivative(self, Z, alpha=0.03):
        self.output=np.where(Z > 0, 1, alpha)
        return self.output

    #Sigmoid function ativazione
    def activation_Sigmoid(self, Z):
        self.output= 1 / (1 + np.exp(-Z))
        return self.output

    def activation_Sigomid_derivative(self, Z):
        s=Funzioni_attivazione.activation_Sigmoid(Z)
        self.output= s * (1-s)
        return self.output

    #Tanh function ativazione
    def activation_tanh(self, Z):
        self.output=np.sinh(Z)/np.cosh(Z)
        return self.output

    def activation_tanh_derivative(self, Z):
        return 1-(Funzioni_attivazione.activation_tanh(Z) ** 2)


class Funzioni_Perdita:
    def __init__(self, y_pred, y_target):
        self.operazione="Perdita"
        self.input_pred=y_pred
        self.input_target=y_target
        self.output=None
        
    def func(self, y_pred, y_target, type, derivata):
        if type == "MAE":
            if (not derivata):
                return self.Loss_MAE(y_label=self.input_target, y_pred=self.input_pred)
            else:
                return self.Loss_MAE_derivative(y_label=self.input_target, y_pred=self.input_pred)
        elif type == "MSE":
            if (not derivata):
                return self.Loss_MSE(y_label=self.input_target, y_pred=self.input_pred)
            else: 
                return self.Loss_MSE_derivative(y_label=self.input_target, y_pred=self.input_pred)
        elif type == "BCE":
            if (not derivata):
                return self.Loss_BCE(y_label=self.input_target, y_pred=self.input_pred)
            else:
                return self.Loss_BCE_derivative(y_label=self.input_target, y_pred=self.input_pred)
        elif type == "CCE":
            if(not derivata):
                return self.Loss_CCE(y_label=self.input_target, y_pred=self.input_pred)
            else:
                pass
        else:
            raise ValueError(f"funzione di costo {type}, non supportata")

 #mse Loss
    def Loss_MSE(y_pred, y_label):
        return np.mean((y_pred-y_label)**2)
        
    def Loss_MSE_derivative(y_pred, y_label):
        n=len(y_label)
        y_label=y_label.reshape(-1, 1)
        MSE_derivata=-2 * (y_pred-y_label) / n
        return MSE_derivata

    #MAE Loss
    def Loss_MAE(y_pred, y_label):
        return np.abs(np.mean(y_pred-y_label))

    def Loss_MAE_derivative(y_pred, y_label):
        n = len(y_label)
        y_label=y_label.reshape(-1, 1)
        return np.where(y_pred < y_label, -1/n, 1/n)    

    #Binary Cross Entropy Loss
    def Loss_BCE(y_pred, y_label):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.mean(y_label * np.log(y_pred) + (1 - y_label) * np.log(1 - y_pred))
        return loss

    def Loss_BCE_derivative(y_pred, y_label):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        derivative = -(y_label / y_pred) + (1 - y_label) / (1 - y_pred)
        return derivative
    
    def Loss_CCE(y_pred, y_label):
        eps = 1e-15  # evita log(0)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss=-np.sum(y_label * np.log(y_pred))
        return loss

  

class nn_optimizers:
    def __init__(self):
        pass

    #gli algoritmi di otimizazzione per addestramento dei pesi
    def optimizer_SGD(layers:list, attivazzioni:list, somme_pesate:list, features, targets:np.ndarray, pesi, bias, lr, activation_fn, loss_fn, legge_fisica):
        gradiente_pesi=[np.ones_like(p) for p in pesi]
        gradiente_bias=[np.ones_like(b) for b in bias]
        
        for layer in reversed(range(len(layers))):
            attivazzione_corrente=attivazzioni[layer]
            attivazzione_pregressa=attivazzioni[layer-1] 

            #print(f"\nretroprogazione strato: {layer}")
            if layer == (len(layers)-1):
                derivata_loss_fisica=Fisica.MAE_derivata_leggeCoulomb(Fisica, features=features, y_pred=attivazzione_corrente)
                derivata_loss_data=nn_functions.Loss(nn_functions, y_pred=attivazzione_corrente, y_target=targets, type=loss_fn, derivata=True)
                derivata_attivazione=nn_functions.activation(nn_functions, Z=somme_pesate[layer], type=activation_fn, derivata=True)

                #print(f"derivata loss data: {derivata_loss_data.shape}")
                #print(f"derivata attivazione: {derivata_attivazione.shape}")
                #print(f"derivata loss fisica: {derivata_loss_fisica.shape}")
                
                lambda_fisica=1e-6
                delta_output=(derivata_loss_data + derivata_loss_fisica * lambda_fisica) * derivata_attivazione

                #print(f"delta output: {delta_output.shape}")

                gradiente_pesi[layer]=np.dot(delta_output.T, attivazzione_pregressa) / len(targets)
                gradiente_bias[layer]=np.sum(delta_output, axis=0, keepdims=True) /len(targets)

                #print(f"gradiente_output: {gradiente_pesi[layer].shape}")
                #print(f"pesi {pesi[layer].shape}")

                pesi[layer] -= lr * gradiente_pesi[layer]
                #bias[layer] -= lr * gradiente_bias[layer]
            else:
                derivata_attivazione=nn_functions.activation(nn_functions, Z=somme_pesate[layer], type=activation_fn ,derivata=1)

                #print(f"derivata attivazione: {derivata_attivazione.shape}")
                #print(f"derivata somma pesata: {derivata_somma_pesata.shape}")
                #print(f"gradiente errore successivo: {gradiente_pesi[layer].shape}")
                delta_output=np.dot(delta_output, pesi[layer + 1]) * derivata_attivazione

                gradiente_pesi[layer]=np.dot(delta_output.T, attivazzione_pregressa) / len(targets)


                #print(f"gradiente_output: {gradiente_pesi[layer]}\n\nshape gradiente output: {gradiente_pesi[layer].shape}")
                #print(f"shape pesi strato {layer} -> {pesi[layer].shape}")
                #print(f"learning rate: {lr}")

                pesi[layer] -= lr * gradiente_pesi[layer]
                #bias[layer] -= lr * gradiente_bias[layer]





from src.Tools.PIML import Fisica
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

class SommaPesata:
    def __init__(self, X_inputs, bias, pesi):
        self.operazione="somma_pesata"
        self.in_features=np.array(X_inputs)
        self.pesi=pesi
        self.bias=bias
        self.out_features=None

    def func(self, strato, derivata):
        if not derivata:
            return self.nn_SommaPesata(layer=strato)
        else:    
            return self.nn_derivata_sommaPesata(layer=strato)

    def nn_SommaPesata(self, layer:int):
        print(f"pesi type: {type(self.pesi[layer])} features type: {type(self.in_features)} bias type: {type(self.bias[layer])}")
        print(f"pesi shape: {self.pesi[layer].T.shape} features shape: {self.in_features.shape} bias shape: {self.bias[layer].shape}")
        pesi=self.pesi[layer]
        bias=self.bias[layer]
        self.out_features=np.matmul(pesi.T, self.in_features) + bias
        print(f"out_features shape: {self.out_features.shape}")
        return self.out_features
            
    
    def nn_derivata_sommaPesata(self, layer):
        pesi=self.pesi[layer]
        #print(f"pesi transposed shape: {pesi.T.shape}")
        return pesi.T


class attivazione:
    def __init__(self):
        self.operazione="attivazione"
        self.input_Z=None
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
        print(f"ativazzione: {type(Z)}")
        print(f"attivazione: {Z.shape}")
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
        s=attivazione.activation_Sigmoid(Z)
        self.output= s * (1-s)
        return self.output

    #Tanh function ativazione
    def activation_tanh(self, Z):
        self.output=np.sinh(Z)/np.cosh(Z)
        return self.output

    def activation_tanh_derivative(self, Z):
        return 1-(attivazione.activation_tanh(Z) ** 2)


class Perdita:
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
    def Loss_MSE(self, y_pred, y_label):
        self.output=np.mean((y_pred-y_label)**2)
        return self.output
        
    def Loss_MSE_derivative(self, y_pred, y_label):
        n=len(y_label)
        y_label=y_label.reshape(-1, 1)
        self.output=-2 * (y_pred-y_label) / n
        return self.output

    #MAE Loss
    def Loss_MAE(self, y_pred, y_label):
        self.output=np.abs(np.mean(y_pred-y_label))
        return self.output

    def Loss_MAE_derivative(self, y_pred, y_label):
        n = len(y_label)
        y_label=y_label.reshape(-1, 1)
        self.output=np.where(y_pred < y_label, -1/n, 1/n)    
        return self.output

    #Binary Cross Entropy Loss
    def Loss_BCE(self, y_pred, y_label):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        self.output=-np.mean(y_label * np.log(y_pred) + (1 - y_label) * np.log(1 - y_pred))
        return self.output

    def Loss_BCE_derivative(self, y_pred, y_label):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        self.output=-(y_label / y_pred) + (1 - y_label) / (1 - y_pred)
        return self.output
    
    def Loss_CCE(self, y_pred, y_label):
        eps = 1e-15  # evita log(0)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        self.output=-np.sum(y_label * np.log(y_pred))
        return self.output

  

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





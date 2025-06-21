from src.Tools.PIML import Fisica
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

class SommaPesata:
    def __init__(self, bias, pesi):
        self.operazione="somma_pesata"
        self.pesi=pesi
        self.bias=bias

    def func(self, inputs, strato, derivata):
        if not derivata:
            return self.nn_SommaPesata(layer=strato, inputs=inputs)
        else:    
            return self.nn_derivata_sommaPesata(layer=strato)

    def nn_SommaPesata(self, inputs, layer:int):
        print(f"features type: {type(inputs.shape)} pesi type: {type(self.pesi[layer])} bias type: {type(self.bias[layer])}")
        print(f"features shape: {inputs.shape} pesi shape: {self.pesi[layer].T.shape} bias shape: {self.bias[layer].shape}")
        pesi=self.pesi[layer]
        bias=self.bias[layer]
        out_features=np.matmul(inputs, pesi.T) + bias
        print(f"out_features shape: {out_features.shape}")
        return out_features
            
    
    def nn_derivata_sommaPesata(self, layer):
        print(f"pesi shape: {self.pesi[layer].shape}")
        pesi=self.pesi[layer]
        return pesi


class attivazione:
    def __init__(self, type):
        self.operazione="attivazione"
        self.type=type

    def func(self, inputs, type:str, derivata:bool):
        if type == "ReLU":
            if (not derivata):
                return self.activation_ReLU(Z=inputs)
            else:
                return self.activation_ReLU_derivative(Z=inputs)
        elif type == "leaky_ReLU":
            if(not derivata):
                return self.activation_leaky_ReLU(Z=inputs)
            else:
                return self.activation_leaky_ReLU_derivative(Z=inputs)
        elif type == "Sigmoid":
            if (not derivata):
                return self.activation_Sigmoid(Z=inputs)
            else: 
                return self.activation_Sigomid_derivative(Z=inputs)
        elif type == "Tanh":
            if (not derivata):
                return self.activation_tanh(Z=inputs)
            else: 
                return self.activation_tanh_derivative(Z=inputs)
        else:
            raise ValueError(f"la funzione {type} non e supportata")

    #ReLU function ativazione
    def activation_ReLU(self, Z):
        result=np.maximum(0, Z)
        return result

    def activation_ReLU_derivative(self, Z):
        output=np.where(Z > 0, 1, 0)
        return output

    #Leaky ReLU variant ativazione
    def activation_leaky_ReLU(self, Z, alpha=0.03):
        print(f"attivazione shape: {Z.shape}")
        output=np.where(Z >= 0, Z, alpha * Z)
        return output

    def activation_leaky_ReLU_derivative(self, Z, alpha=0.03):
        print(f"attivazione shape: {Z.shape}")
        output=np.where(Z > 0, 1, alpha)
        return output

    #Sigmoid function ativazione
    def activation_Sigmoid(self, Z):
        output= 1 / (1 + np.exp(-Z))
        return output

    def activation_Sigomid_derivative(self, Z):
        s=self.activation_Sigmoid(Z)
        output= s * (1-s)
        return output

    #Tanh function ativazione
    def activation_tanh(self, Z):
        output=np.sinh(Z)/np.cosh(Z)
        return output

    def activation_tanh_derivative(self, Z):
        return 1-(self.activation_tanh(Z) ** 2)


class Perdita:
    def __init__(self, type):
        self.operazione="Perdita"
        self.type=type
        
    def func(self, y_pred, y_target, derivata, type=type):
        if type == "MAE":
            if (not derivata):
                return self.Loss_MAE(y_label=y_target, y_pred=y_pred)
            else:
                return self.Loss_MAE_derivative(y_label=y_target, y_pred=y_pred)
        elif type == "MSE":
            if (not derivata):
                return self.Loss_MSE(y_label=y_target, y_pred=y_pred)
            else: 
                return self.Loss_MSE_derivative(y_label=y_target, y_pred=y_pred)
        elif type == "BCE":
            if (not derivata):
                return self.Loss_BCE(y_label=y_target, y_pred=y_pred)
            else:
                return self.Loss_BCE_derivative(y_label=y_target, y_pred=y_pred)
        elif type == "CCE":
            if(not derivata):
                return self.Loss_CCE(y_label=y_target, y_pred=y_pred)
            else:
                pass
        else:
            raise ValueError(f"funzione di costo {type}, non supportata")

 #mse Loss
    def Loss_MSE(self, y_pred, y_label):
        print(f"preds: {y_pred.shape} targets: {y_label.shape}")
        output=np.mean((y_pred-y_label)**2)
        return output
        
    def Loss_MSE_derivative(self, y_pred, y_label):
        n=len(y_label)
        y_label=y_label.reshape(-1, 1)
        print(f"y_pred: {y_pred.shape} y_label: {y_label.shape}")
        output=-2 * (y_pred-y_label) / n
        print(f"output: {output.shape}")
        return output

    #MAE Loss
    def Loss_MAE(self, y_pred, y_label):
        output=np.abs(np.mean(y_pred-y_label))
        return output

    def Loss_MAE_derivative(self, y_pred, y_label):
        n = len(y_label)
        y_label=y_label.reshape(-1, 1)
        output=np.where(y_pred < y_label, -1/n, 1/n)    
        return output

    #Binary Cross Entropy Loss
    def Loss_BCE(self, y_pred, y_label):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        output=-np.mean(y_label * np.log(y_pred) + (1 - y_label) * np.log(1 - y_pred))
        return output

    def Loss_BCE_derivative(self, y_pred, y_label):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        output=-(y_label / y_pred) + (1 - y_label) / (1 - y_pred)
        return output
    
    def Loss_CCE(self, y_pred, y_label):
        eps = 1e-15  # evita log(0)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        output=-np.sum(y_label * np.log(y_pred))
        return output

  

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





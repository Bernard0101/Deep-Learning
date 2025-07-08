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
        #print(f"features type: {type(inputs.shape)} pesi type: {type(self.pesi[layer])} bias type: {type(self.bias[layer])}")
        #print(f"features shape: {inputs.shape} pesi shape: {self.pesi[layer].T.shape} bias shape: {self.bias[layer].shape}")
        pesi=self.pesi[layer]
        bias=self.bias[layer]
        out_features=np.matmul(inputs, pesi.T) + bias
        print(f"out_features shape: {out_features.shape}")
        return out_features
            
    
    def nn_derivata_sommaPesata(self, layer):
        #print(f"pesi shape: {self.pesi[layer].shape}")
        pesi=self.pesi[layer]
        return pesi.T


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
        #print(f"attivazione shape: {Z.shape}")
        output=np.where(Z >= 0, Z, alpha * Z)
        return output

    def activation_leaky_ReLU_derivative(self, Z, alpha=0.03):
        #print(f"attivazione shape: {Z.shape}")
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
                return self.Loss_CCE_derivative(y_label=y_target, y_pred=y_pred)
        else:
            raise ValueError(f"funzione di costo {type}, non supportata")

 #mse Loss
    def Loss_MSE(self, y_pred, y_label):
        #print(f"preds: {y_pred.shape} targets: {y_label.shape}")
        return np.mean((y_pred-y_label)**2)
        
    def Loss_MSE_derivative(self, y_pred, y_label):
        n=len(y_label)
        y_label=y_label.reshape(-1, 1)
        #print(f"y_pred: {y_pred.shape} y_label: {y_label.shape}")
        return -2 * (y_pred-y_label) / n
        #print(f"output: {output.shape}")

    #MAE Loss
    def Loss_MAE(self, y_pred, y_label):
        return np.abs(np.mean(y_pred-y_label))

    def Loss_MAE_derivative(self, y_pred, y_label):
        n=len(y_label)
        y_label=y_label.reshape(-1, 1)
        return np.where(y_pred < y_label, -1/n, 1/n)    

    #Binary Cross Entropy Loss
    def Loss_BCE(self, y_pred, y_label):
        y_pred=np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_label * np.log(y_pred) + (1 - y_label) * np.log(1 - y_pred))

    def Loss_BCE_derivative(self, y_pred, y_label):
        y_pred=np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(y_label / y_pred) + (1 - y_label) / (1 - y_pred)
    
    def Loss_CCE(self, y_pred, y_label):
        eps=1e-15  # evita log(0)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.sum(y_label * np.log(y_pred))
    
    def Loss_CCE_derivative(self, y_pred, y_label):
        return -np.sum(y_label/y_pred)

  

class optimizers:
    def __init__(self, alg_optim, grad_pesi, grad_bias):
        self.optim=alg_optim
        self.gradiente_pesi=grad_pesi
        self.gradiente_bias=grad_bias

    def func(self, pesi, bias, lr, type):
        if type == "SGD":
            self.optimizer_SGD(pesi=pesi, bias=bias, lr=lr)
        else:
            raise ValueError(f"ottimizzattore: {type} non supportato")
            

    #gli algoritmi di otimizazzione per addestramento dei pesi
    def optimizer_SGD(self, pesi, bias, lr):
       for i in reversed(range(len(pesi))):
           #print(f"pesi: {pesi[i].shape} grad pesi: {self.gradiente_pesi[i].T.shape}")
           pesi[i] -= self.gradiente_pesi[i].T * lr
           bias[i] -= self.gradiente_bias[i].T * lr
    
    def optimizer_Adagrad(self, pesi, bias):
        pass
        




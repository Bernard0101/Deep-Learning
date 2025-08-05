import numpy as np # type: ignore


class SommaPesata:
    def __init__(self, autodiff, bias, pesi):
        self.autodiff=autodiff
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
        #print(f"features shape: {inputs.shape} pesi shape: {self.pesi[layer].shape} bias shape: {self.bias[layer].shape}")
        pesi=self.pesi[layer]
        bias=self.bias[layer]
        out_features=np.matmul(inputs, pesi) + bias
        #print(f"out_features shape: {out_features.shape}")
        self.autodiff.memorizzare(strato=layer, inputs=inputs, outputs=out_features, operazione="somma_pesata")
        return out_features
            
    
    def nn_derivata_sommaPesata(self, layer):
        #print(f"pesi shape: {self.pesi[layer].shape}")
        pesi=self.pesi[layer]
        return pesi.T


class attivazione:
    def __init__(self, autodiff, type):
        self.autodiff=autodiff
        self.operazione="attivazione"
        self.type=type

    def func(self, inputs, strato, derivata):
        if self.type == "ReLU":
            if (not derivata):
                return self.activation_ReLU(Z=inputs, layer=strato)
            else:
                return self.activation_ReLU_derivative(Z=inputs)
        elif self.type == "leaky_ReLU":
            if(not derivata):
                return self.activation_leaky_ReLU(Z=inputs, layer=strato)
            else:
                return self.activation_leaky_ReLU_derivative(Z=inputs)
        elif self.type == "Sigmoid":
            if (not derivata):
                return self.activation_Sigmoid(Z=inputs, layer=strato)
            else: 
                return self.activation_Sigomid_derivative(Z=inputs, layer=strato)
        elif self.type == "Tanh":
            if (not derivata):
                return self.activation_tanh(Z=inputs, layer=strato)
            else: 
                return self.activation_tanh_derivative(Z=inputs, layer=strato)
        else:
            raise ValueError(f"la funzione: {self.type}, non e supportata")

    #ReLU function ativazione
    def activation_ReLU(self, Z, layer):
        output=np.maximum(0, Z)
        self.autodiff.memorizzare(strato=layer, inputs=Z, outputs=output, operazione="attivazione")
        return output

    def activation_ReLU_derivative(self, Z):
        output=np.where(Z > 0, 1, 0)
        return output

    #Leaky ReLU variant ativazione
    def activation_leaky_ReLU(self, Z, layer, alpha=0.03):
        #print(f"attivazione shape: {Z.shape}")
        output=np.where(Z >= 0, Z, alpha * Z)
        self.autodiff.memorizzare(strato=layer, inputs=Z, outputs=output, operazione="attivazione")
        return output

    def activation_leaky_ReLU_derivative(self, Z, alpha=0.03):
        #print(f"attivazione shape: {Z.shape}")
        output=np.where(Z > 0, 1, alpha)
        return output

    #Sigmoid function ativazione
    def activation_Sigmoid(self, Z, layer):
        output= 1 / (1 + np.exp(-Z))
        self.autodiff.memorizzare(strato=layer, inputs=Z, outputs=output, operazione="attivazione")
        return output

    def activation_Sigomid_derivative(self, Z, layer):
        s=self.activation_Sigmoid(Z, layer=layer)
        output= s * (1-s)
        return output

    #Tanh function ativazione
    def activation_tanh(self, Z, layer):
        output=np.sinh(Z)/np.cosh(Z)
        self.autodiff.memorizzare(strato=layer, inputs=Z, outputs=output, operazione="attivazione")
        return output

    def activation_tanh_derivative(self, Z, layer):
        return 1-(self.activation_tanh(Z, layer) ** 2)


class Perdita:
    def __init__(self, autodiff, type):
        self.autodiff=autodiff
        self.operazione="Perdita"
        self.type=type
        
    def func(self, y_pred, y_target, derivata):
        if self.type == "MAE":
            if (not derivata):
                return self.Loss_MAE(y_label=y_target, y_pred=y_pred)
            else:
                return self.Loss_MAE_derivative(y_label=y_target, y_pred=y_pred)
        elif self.type == "MSE":
            if (not derivata):
                return self.Loss_MSE(y_label=y_target, y_pred=y_pred)
            else: 
                return self.Loss_MSE_derivative(y_label=y_target, y_pred=y_pred)
        elif self.type == "BCE":
            if (not derivata):
                return self.Loss_BCE(y_label=y_target, y_pred=y_pred)
            else:
                return self.Loss_BCE_derivative(y_label=y_target, y_pred=y_pred)
        elif self.type == "CCE":
            if(not derivata):
                return self.Loss_CCE(y_label=y_target, y_pred=y_pred)
            else:
                return self.Loss_CCE_derivative(y_label=y_target, y_pred=y_pred)
        else:
            raise ValueError(f"funzione di costo: {self.type}, non e supportata")

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
    def __init__(self, autodiff, alg_optim, pesi, bias):
        self.autodiff=autodiff
        self.optim=alg_optim
        self.gradiente_pesi=self.autodiff.gradiente_pesi
        self.gradiente_bias=self.autodiff.gradiente_bias
        self.grad_pesi_anteriore=[np.zeros_like(p)for p in pesi]
        self.grad_bias_anteriore=[np.zeros_like(b)for b in bias]
        self.moving_average_gradiente_pesi=[np.zeros_like(p)for p in pesi]
        self.moving_average_gradiente_bias=[np.zeros_like(b)for b in bias]

    def algorithm(self, pesi, bias, lr):
        if self.optim == "SGD":
            self.optimizer_SGD(pesi=pesi, bias=bias, lr=lr)
        elif self.optim == "Adagrad":
            self.optimizer_Adagrad(pesi=pesi, bias=bias, lr=lr)
        elif self.optim == "RMSprop":
            self.optimizer_RMSprop(pesi=pesi, bias=bias, lr=lr, beta=0.95)
        else:    
            raise ValueError(f"ottimizzattore: {self.optim} non supportato")
            

    #gli algoritmi di otimizazzione per addestramento dei pesi
    def optimizer_SGD(self, pesi, bias, lr):
       for i in reversed(range(len(pesi))):
           #print(f"pesi: {pesi[i].shape} grad pesi: {self.gradiente_pesi[i].T.shape}")
           pesi[i] -= self.gradiente_pesi[i] * lr
           bias[i] -= self.gradiente_bias[i] * lr
    

    def optimizer_Adagrad(self, pesi, bias, lr):
        for i in reversed(range(len(pesi))):
            gradiente_pesi_accumulato=self.grad_pesi_anteriore[i] + np.power(self.gradiente_pesi[i], 2)
            gradiente_bias_accumulato=self.grad_bias_anteriore[i] + np.power(self.gradiente_bias[i], 2)
            #print(f"grad_pesi accumulato: {gradiente_pesi_accumulato.shape} grad_pesi: {self.gradiente_pesi[i].shape}")
            #print(f"grad_bias accumulato: {gradiente_bias_accumulato.shape} grad_bias: {self.gradiente_bias[i].shape}")
            #print(f"pesi: {pesi[i].shape} grad_pesi_accumulato: {gradiente_pesi_accumulato[i].shape}")
            #print(f"bias: {bias[i].shape} grad_bias_accumulato: {gradiente_bias_accumulato[i].shape}")
            pesi[i] -= lr / np.sqrt(gradiente_pesi_accumulato + 3e-9) * self.gradiente_pesi[i]
            bias[i] -= lr / np.sqrt(gradiente_bias_accumulato + 3e-9) * self.gradiente_bias[i]
            self.grad_pesi_anteriore[i]=gradiente_pesi_accumulato
            self.grad_bias_anteriore[i]=gradiente_bias_accumulato


    def optimizer_RMSprop(self, pesi, bias, lr, beta):
        for i in reversed(range(len(pesi))):
            #print(f"moving_average_pesi: {self.moving_average_gradiente_pesi[i].shape} gradiente_pesi: {self.gradiente_pesi[i].shape}")
            #print(f"moving_average_bias: {self.moving_average_gradiente_bias[i].shape} gradiente_bias: {self.gradiente_bias[i].shape}")
            moving_average_gradiente_pesi=beta * self.moving_average_gradiente_pesi[i] + (1-beta) * np.power(self.gradiente_pesi[i], 2)
            moving_average_gradiente_bias=beta * self.moving_average_gradiente_bias[i] + (1-beta) * np.power(self.gradiente_bias[i], 2)
            pesi[i] -= lr / np.sqrt(moving_average_gradiente_pesi + 3e-9) * self.gradiente_pesi[i]
            bias[i] -= lr / np.sqrt(moving_average_gradiente_bias + 3e-9) * self.gradiente_bias[i]
            self.moving_average_gradiente_pesi[i]=moving_average_gradiente_pesi
            self.moving_average_gradiente_bias[i]=moving_average_gradiente_bias

            


        




import numpy as np # type: ignore


class nn_functions:
    def __init__(self):
        pass

    def activation(self, type:str, derivata:bool, Z:np.ndarray):
        if type == "ReLU":
            if (not derivata):
                return self.activation_ReLU(Z=Z)
            else:
                return self.activation_ReLU_derivative(Z=Z)
        elif type == "leaky_ReLU":
            if(not derivata):
                return self.activation_leaky_ReLU(Z=Z)
            else:
                return self.activation_leaky_ReLU_derivative(Z=Z)
        elif type == "Sigmoid":
            if (not derivata):
                return self.activation_Sigmoid(Z=Z)
            else: 
                return self.activation_Sigomid_derivative(Z=Z)
        elif type == "Tanh":
            if (not derivata):
                return self.activation_tanh(Z=Z)
            else: 
                return self.activation_tanh_derivative(Z=Z)
        else:
            raise ValueError(f"la funzione {type} non e supportata")

    #ReLU function ativazione
    def activation_ReLU(Z):
        return np.maximum(0, Z)

    def activation_ReLU_derivative(Z):
        return np.where(Z > 0, 1, 0)

    #Leaky ReLU variant ativazione
    def activation_leaky_ReLU(Z, alpha=0.03):
        return np.where(Z >= 0, Z, alpha * Z)

    def activation_leaky_ReLU_derivative(Z, alpha=0.03):
        return np.where(Z > 0, 1, alpha)

    #Sigmoid function ativazione
    def activation_Sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

    def activation_Sigomid_derivative(Z):
        s = nn_functions.activation_Sigmoid(Z)
        return s * (1-s)

    #Tanh function ativazione
    def activation_tanh(Z):
        return np.sinh(Z)/np.cosh(Z)

    def activation_tanh_derivative(Z):
        return 1-(nn_functions.activation_tanh(Z) ** 2)

    def Loss(self, y_pred, y_target, type, derivata):
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
        else:
            raise ValueError(f"funzione di costo {type}, non supportata")

    #mse Loss
    def Loss_MSE(y_pred, y_label):
        return np.mean((y_pred-y_label)**2)
        
    def Loss_MSE_derivative(y_pred, y_label):
        return np.mean(2 * (y_pred-y_label))

    #MAE Loss
    def Loss_MAE(y_pred, y_label):
        return np.abs(np.mean(y_pred-y_label))

    def Loss_MAE_derivative(y_pred, y_label):
        n = len(y_label)
        gradients = np.where(y_pred < y_label, -1 / n, 1 / n)
        gradients[y_pred == y_label] = 0 
        return gradients    

    #Binary Cross Entropy Loss
    def Loss_BCE(y_pred, y_label):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.mean(y_label * np.log(y_pred) + (1 - y_label) * np.log(1 - y_pred))
        return loss

    def Loss_BCE_derivative(y_pred, y_label):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        derivative = -(y_label / y_pred) + (1 - y_label) / (1 - y_pred)
        return derivative

    def Loss_Softmax(Z):
        exp_z=np.exp(Z - np.max(Z)) 
        return exp_z / np.sum(exp_z)

    def Loss_Softmax_derivative(Z):
        s=nn_functions.Loss_Softmax(Z).reshape(-1, 1) 
        return np.diagflat(s) - np.dot(s, s.T) 

class nn_optimizers:
    def __init__(self):
        pass

    #gli algoritmi di otimizazzione per addestramento dei pesi
    def optimizer_SGD(layers:list, attivazzioni:list, somme_pesate:list, targets:np.ndarray, pesi, bias, lr, activation_fn, loss_fn):
        gradiente_pesi=[np.ones_like(p) for p in pesi]
        gradiente_bias=[np.ones_like(b) for b in bias]
        
        for layer in reversed(range(len(layers))):
            attivazzione_corrente=attivazzioni[layer]
            attivazzione_seguente=attivazzioni[layer-1] 

            #print(f"\nretroprogazione strato: {layer}")
            if layer == (len(layers)-1):    
                derivata_perdita=nn_functions.Loss(nn_functions, y_pred=attivazzione_corrente, y_target=targets, type=loss_fn, derivata=1)
                derivata_attivazione=nn_functions.activation(nn_functions, Z=somme_pesate[layer], type=activation_fn, derivata=1)
                derivata_somma_pesata=attivazzione_corrente

                #print(f"derivata perdita: {derivata_perdita}")
                #print(f"derivata attivazione: {derivata_attivazione.shape}")
                #print(f"derivata somma pesata: {derivata_somma_pesata.shape}")

                gradiente_pesi[layer]=np.dot(derivata_attivazione.T, derivata_somma_pesata) * derivata_perdita
                gradiente_bias[layer]=np.dot(derivata_attivazione.T, derivata_somma_pesata) * derivata_perdita

                #print(f"gradiente_output: {gradiente_pesi[layer]}")

                pesi[layer] -= lr * gradiente_pesi[layer]
                #bias[layer] -= lr * gradiente_bias[layer]
            else:
                derivata_attivazione=nn_functions.activation(nn_functions, Z=somme_pesate[layer], type=activation_fn ,derivata=1)
                derivata_somma_pesata=attivazzione_seguente

                #print(f"derivata attivazione: {derivata_attivazione.shape}")
                #print(f"derivata somma pesata: {derivata_somma_pesata.shape}")
                #print(f"gradiente errore successivo: {gradiente_pesi[layer].shape}")

                gradiente_pesi[layer]=np.dot(derivata_somma_pesata.T, derivata_attivazione) * gradiente_pesi[layer].T

                #print(f"gradiente_output: {gradiente_pesi[layer]}\n\nshape gradiente output: {gradiente_pesi[layer].shape}")
                #print(f"shape pesi strato {layer} -> {pesi[layer].shape}")
                #print(f"learning rate: {lr}")

                pesi[layer] -= lr * gradiente_pesi[layer].T
                #bias[layer] -= lr * gradiente_bias[layer]




    def optimizer_Adagrad(layers, attivazioni, targets, pesi, bias, lr, e=1e-8):
        cache_gradienti_pesi=[np.zeros_like(p) for p in pesi]
        cache_gradienti_bias=[np.zeros_like(b) for b in bias]
        for layer in reversed(range(len(layers))):
            attivazione_pregressa=attivazioni[layer-1]
            attivazione_corrente=attivazioni[layer]

            if layer == len(layers):
                derivata_errore=nn_functions.Loss_MSE_derivative(y_pred=attivazione_corrente, y_label=targets)
            else:
                derivata_errore=nn_functions.Loss_MSE_derivative(y_pred=attivazione_pregressa.T, y_label=targets)
            
            derivata_attivazione=nn_functions.activation_leaky_ReLU_derivative(Z=attivazione_corrente)

            gradiente=derivata_attivazione * derivata_errore

            gradiente_pesi=np.dot(gradiente.T, attivazione_pregressa) / len(targets)
            gradiente_bias=np.sum(gradiente, axis=0) / len(targets)

            cache_gradienti_pesi[layer] += (gradiente_pesi ** 2)
            cache_gradienti_bias[layer] += (gradiente_bias ** 2)

            pesi[layer] -= (lr / np.sqrt(cache_gradienti_pesi[layer] + e) * gradiente_pesi)
            bias[layer] -= (lr / np.sqrt(cache_gradienti_bias[layer] + e) * gradiente_bias.reshape(-1))

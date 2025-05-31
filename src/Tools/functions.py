import numpy as np # type: ignore

class nn_functions:
    def __init__(self):
        pass
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
    def optimizer_SGD(layers, attivazzioni, targets, pesi, bias, lr):
        for layer in reversed(range(len(layers))):
            attivazzione_pregressa=attivazzioni[layer-1]
            attivazzione_corrente=attivazzioni[layer]

            derivata_errore=nn_functions.Loss_MSE_derivative(y_pred=attivazzione_pregressa.T, y_label=targets)

            derivata_attivazione=nn_functions.activation_tanh_derivative(Z=attivazzione_corrente)

            gradiente=derivata_attivazione * derivata_errore

            gradiente_pesi=np.dot(gradiente.T, attivazzione_pregressa) / len(targets)
            gradiente_bias=np.sum(gradiente, axis=0, keepdims=True) / len(targets)

            pesi[layer] -= lr * gradiente_pesi
            bias[layer] -= lr * gradiente_bias.reshape(-1)


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

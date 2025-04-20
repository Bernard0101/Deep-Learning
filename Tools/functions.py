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
        return np.mean(y_pred-y_label)

    def Loss_MAE_derivative(y_pred, y_label):
        n = len(y_label)
        gradients = np.where(y_pred < y_label, -1 / n, 1 / n)
        gradients[y_pred == y_label] = 0 
        return gradients    

    #Binary Cross Entropy Loss
    def Loss_Binary_Cross_Entropy(y_pred, y_label):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.mean(y_label * np.log(y_pred) + (1 - y_label) * np.log(1 - y_pred))
        return loss

    def Loss_Binary_Cross_Entropy_derivative(y_pred, y_label):
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
    def optimizer_SGD(layers, ativazzioni, labels, pesi, bias, lr, ativazione):
        for layer in reversed(range(len(layers))):
            ativazzione_pregressa=ativazzioni[layer-1]
            ativazzione_corrente=ativazzioni[layer]

            derivata_errore=nn_functions.Loss_MSE_derivative(y_pred=ativazzione_pregressa.T, y_label=labels)

            derivata_ativazione=nn_functions.activation_leaky_ReLU_derivative(Z=ativazzione_corrente)

            gradiente=derivata_ativazione * derivata_errore

            gradiente_pesi=np.dot(ativazzione_pregressa.T, gradiente) / len(labels)
            gradiente_bias=np.sum(gradiente, axis=0, keepdims=True) / len(labels)

            pesi[layer] -= lr * gradiente_pesi.T
            bias[layer] -= lr * gradiente_bias.reshape(-1)


    def optimizer_Momentum():
        pass
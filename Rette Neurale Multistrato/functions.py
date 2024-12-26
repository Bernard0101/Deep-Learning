import numpy as np

class nn_functions:
    def __init__(self):
        pass

    #ReLU function ativazione
    def activation_ReLU(Z):
        return np.maximum(0, Z)

    def activation_ReLU_derivative(Z):
        return np.where(Z > 0, 1, 0)

    #Leaky ReLU variant ativazione
    def activation_leaky_ReLU(Z, alpha=0.01):
        return np.where(Z >= 0, Z, alpha * Z)
    
    def activation_leaky_ReLU_derivative(Z, alpha=0.01):
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
        exp_z = np.exp(Z - np.max(Z)) 
        return exp_z / np.sum(exp_z)

    def Loss_Softmax_derivative(Z):
        s = nn_functions.Loss_Softmax(Z).reshape(-1, 1) 
        return np.diagflat(s) - np.dot(s, s.T) 

    #le algoritmi di otimizazzione per aggiornamento dei pesi
    def optimizer_SGD(self, layers, ativazzioni, labels, pesi, bias, learning_rate):
        for layer in reversed(range(len(layers))):
            layer_ativazioni_indietro=ativazzioni[layer-1]
            layer_ativazione=ativazzioni[layer]

            #derivata a rispeto della funzione di perdita
            derivata_errore=self.Loss_MSE_derivative(layer_ativazioni_indietro.T, labels)

            #derivata a rispeto della funzione de ativazzione
            derivata_ativazione=self.activation_leaky_ReLU_derivative(layer_ativazione)

            #regola della cattena
            gradiente=derivata_ativazione * derivata_errore

            #adesso fare il calcolo del gradiente a rispeto di ogni pesi e bias 
            derivata_pesi=np.dot(layer_ativazioni_indietro.T, gradiente)
            derivata_bias=np.sum(gradiente, axis=0, keepdims=True)
            
            #aggiornamento dei pesi e bias
            pesi[layer] -= learning_rate * derivata_pesi.T
            bias[layer] -= learning_rate * derivata_bias.reshape(-1)
    
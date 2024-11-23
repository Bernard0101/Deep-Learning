import numpy as np

class nn_functions:
    def __init__(self):
        pass

    #ReLU function
    def activation_ReLU(Z):
        return np.maximum(0, Z)

    def activation_ReLU_derivative(Z):
        return np.where(Z > 0, 1, 0)

    #Leaky ReLU variant
    def activation_leaky_ReLU(Z, alpha=0.01):
        return np.where(Z >= 0, Z, alpha * Z)

        #the derivative of leaky ReLU
    def activation_leaky_ReLU_derivative(Z, alpha=0.01):
        return np.where(Z > 0, 1, alpha)


    #mse Loss
    def Loss_MSE(y_pred, y_label):
        return np.mean((y_pred-y_label)**2)
        
        #the derivative of mse loss
    def Loss_MSE_derivative(y_pred, y_label):
        return np.mean(2 * (y_pred-y_label))

    def Loss_MAE(y_pred, y_label):
        return np.mean(y_pred-y_label)
    


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
    
functions=nn_functions()

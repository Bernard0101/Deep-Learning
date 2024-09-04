import numpy as np

class nn_functions:
    def __init__(self):
        pass

    #ReLU function
    def activation_ReLU(Z):
        return Z if Z >= 0 else 0

    def activation_ReLU_derivative(self, Z):
        return Z if Z>=0 else 0

    #Leaky ReLU variant
    def activation_leaky_ReLU(Z, alpha=0.01):
        return np.where(Z >= 0, Z, alpha * Z)

        #the derivative of leaky ReLU
    def activation_leaky_ReLU_derivative(Z, alpha=0.01):
        return np.where(Z > 0, 1, alpha)


    #mse Loss
    def Loss_MSE(self, y_pred, y_label):
        mse_loss=np.mean((y_pred-y_label)**2)
        return mse_loss
        
        #the derivative of mse loss
    def Loss_MSE_derivative(self, y_pred, y_label):
        return (2 * (y_pred-y_label) / len(y_label))
    
functions=nn_functions()

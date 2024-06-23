import numpy as np
import random

#initialize the perceptron
class Perceptron():
    def __init__(self, learning_rate=0.1, n_features=2):
        self.n_features=n_features
        self.learning_rate=learning_rate
        self.weights=np.random.rand(n_features)
        self.bias=random.uniform(0, 1)

        
    #apply a linear function to all parameters
    def fit(self, features):
        Z=0
        for f in features:
            for x in self.weights:
                Z+= x * f
            Z+=self.bias
            return Z

    #ladder activation funtion
    def activation_function(self, Z):
        return 1 if Z > 1 else 0
    
    #calculate the error for that specific data
    def Calculate_error(self, prediction):
        return labels-prediction



#calculate the loss for each point in the data based on the given prediction
def Calculate_Loss_MSE(loss):
    MSE_error=0
    for index, error in enumerate(loss):
        MSE_error+=(error**2)
    MSE_error=MSE_error/len(loss)
    return MSE_error

                    

#creates the dataset to predict the logic gate and
X=np.array([
           [0,0], 
           [1,0],
           [0,1],
           [1,1]])

y=np.array([0, 0, 0, 1])
    

#crete a training loop for the model 
loss=[]
epochs=100
for features, labels in zip(X, y):
    for epoch in range(epochs):
        perceptron=Perceptron(learning_rate=0.2, n_features=2)

        #apply linear function (ax+b) and an activation funtion (ladder)
        Z=perceptron.fit(features) 
        prediction=perceptron.activation_function(Z)

        #calculate the loss using (MSE)
        error=perceptron.Calculate_error(prediction)
        loss.append(error)
        MSE_loss=Calculate_Loss_MSE(loss)

        #optimize model parameters
        perceptron.weights += perceptron.learning_rate * error * features
        perceptron.bias += perceptron.learning_rate * error

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Mean Loss: {MSE_loss:.4f}")

        







import numpy as np
import random

class Perceptron():
    def __init__(self, learning_rate=0.01, data_length=int, preds=int):
        self.learning_rate=learning_rate
        self.loss=np.random.rand(data_length)
        self.weights=np.random.rand(data_length)
        self.data=np.arange(0, data_length)
        if preds is not None:
            self.predictions=preds
        else:
            self.predictions=np.random.rand(data_length)
        self.bias=random.randint(0, 10)

        
    #aplica uma funcao ax+b em todos as predicoes do modelo
    def fit(self):
        Z=0
        preds=[]
        for index, x in enumerate(self.weights):
            Z+=x*self.predictions[index]
            preds.append(x)
        Z+=self.bias
        return Z, preds

    #usa a funcao degrau, a ativacao mais simples
    def activation_function(self, Z):
        if Z <= 0:
            prediction = 0
            return prediction
        else: 
            prediction = 1
            return prediction
        
    #calculate the loss for each point in the data based on the given prediction
    def Calculate_Loss(self):
        total_loss=0
        for index, i in enumerate(self.data):
            loss=i-self.predictions[index]
            self.loss[index]=loss
            total_loss += loss
        mean_loss=total_loss / len(self.loss)
        return mean_loss
    
    #optimize the model parameters/weights
    def Optimize_model_parameters(self, mean_loss):
        for index, weight in enumerate(self.weights):
            self.weights[index]= weight + (self.learning_rate * self.loss[index]*self.data[index])
        self.bias=self.bias + (self.learning_rate * mean_loss)
        return mean_loss

#crete a training loop for the model 
preds=None
epochs=100
for epoch in range(epochs):
    model_percetron=Perceptron(data_length=10, preds=preds)
    Z, preds=model_percetron.fit()
    prediction=model_percetron.activation_function(Z)
    mean_loss=model_percetron.Calculate_Loss()
    model_percetron.Optimize_model_parameters(mean_loss)
    if epoch % 10 == 0:
     print(f"epoch: {epoch} mean loss: {mean_loss} ")

import numpy as np


def activation_ReLU(Z):
    return Z if Z >= 0 else 0

def calculate_MSE(losses):
    mse_error=0
    for index, error in enumerate(losses):
        mse_error+= error**2
    mse_error/=len(losses)
    return mse_error

class Perceptron:
    def __init__(self, learning_rate=0.001, n_features=0):
        self.features=n_features
        self.learning_rate=learning_rate
        self.weights=np.random.rand(n_features)
        self.loss=np.random.rand(n_features)
        self.bias=np.random.randint(2)

    def activation_ReLU(Z):
        return Z if Z >= 0 else 0

    #aplica funcao de dot product a usa funcao de ativacao ReLU
    def predict(self, X, Z=0):
        Z=0
        for weight in (self.weights):
            for x in X:
                Z+= weight * x
        Z+=self.bias
        prediction=activation_ReLU(Z)
        return prediction
    
    #calculate the error for that specific prediction
    def calculate_error(self, prediction, label):
        loss= label-prediction
        return loss


features=np.array([[0,0],
                   [0,1],
                   [1,0],
                   [1,1]])

labels=np.array([0,0,0,1])


epochs=100
num_features=len(features)
p1=Perceptron(learning_rate=0.01, n_features=num_features)


class TrainPerceptron:
    def __init__(self, model, features, labels):
        self.features=features
        self.labels=labels
        self.model=model
        self.losses=[]
        self.predictions=[]

    def predict(self):
        for epoch in range(epochs):
            losses=[]
            for X, y in zip(features, labels):

                #calculate the sum product and do the activation function
                prediction=self.model.predict(X)

                #calculate the error of that prediction 
                loss=self.model.calculate_error(prediction, label=y)
                losses.append(loss)
                self.losses.append(loss)

                #optimize model parameters
                for x in X:
                    self.model.weights += self.model.learning_rate * loss * x
                    self.model.bias += self.model.learning_rate * loss

            #add the final prediction to the predictions
            self.predictions.append(prediction)
        return self.predictions[-1]



train_perceptron=TrainPerceptron(model=p1, features=features, labels=labels)
preds=train_perceptron.predict()
#print(preds)


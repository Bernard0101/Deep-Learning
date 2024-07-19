import numpy as np

<<<<<<< HEAD

=======
>>>>>>> c38ecb4 (initial neural network skecth yet without backpropagation)
def activation_ReLU(Z):
    return Z if Z >= 0 else 0

def calculate_MSE(losses):
    mse_error=0
    for index, error in enumerate(losses):
        mse_error+= error**2
    mse_error/=len(losses)
    return mse_error

<<<<<<< HEAD
class Percetron:
=======
class Perceptron:
>>>>>>> c38ecb4 (initial neural network skecth yet without backpropagation)
    def __init__(self, learning_rate=0.001, n_features=0):
        self.features= n_features
        self.learning_rate= learning_rate
        self.weights= np.random.rand(n_features)
        self.loss= np.random.rand(n_features)
        self.bias= np.random.randint(2)

<<<<<<< HEAD
=======
    def activation_ReLU(Z):
        return Z if Z >= 0 else 0

>>>>>>> c38ecb4 (initial neural network skecth yet without backpropagation)
    #aplica funcao de dot product a usa funcao de ativacao ReLU
    def fit(self, X, Z=0):
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
<<<<<<< HEAD
p1 = Percetron(n_features = num_features)

class TrainPerceptron:
    def __init__(self, model):
        self.model= model
=======
p1 = Perceptron(n_features = num_features)

class TrainPerceptron:
    def __init__(self, model, features, labels):
        self.features=features
        self.labels=labels
        self.model=model
        self.losses=[]
>>>>>>> c38ecb4 (initial neural network skecth yet without backpropagation)
        self.predictions=[]

    def training_loop(self):
        for epoch in range(epochs):
            losses=[]
<<<<<<< HEAD
            for X, y in zip(features, labels):
=======
            for X, y in zip(self.features, self.labels):
>>>>>>> c38ecb4 (initial neural network skecth yet without backpropagation)

                #calculate the sum product and do the activation function
                prediction=self.model.fit(X)
                self.predictions.append(prediction)

                #calculate the error of that prediction 
                loss=self.model.calculate_error(prediction, label=y)
<<<<<<< HEAD
                losses.append(loss)
=======
                self.losses.append(loss)
>>>>>>> c38ecb4 (initial neural network skecth yet without backpropagation)

                #optimize model parameters
                for x in X:
                    self.model.weights += self.model.learning_rate * loss * x
                self.model.bias += self.model.learning_rate * loss
<<<<<<< HEAD

    
    def predict(self):
        print(self.predictions)
        return self.predictions

train_perceptron= TrainPerceptron(model=p1)
preds=train_perceptron.training_loop()
=======
        return prediction

    
    #def predict(self, preds):
        #print(preds)
        #print(self.predictions)
        #print(self.losses)
        #return self.predictions

train_perceptron= TrainPerceptron(model=p1, features=features, labels=labels)
preds=train_perceptron.training_loop()
#train_perceptron.predict(preds)

>>>>>>> c38ecb4 (initial neural network skecth yet without backpropagation)

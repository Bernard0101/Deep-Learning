import numpy as np # type: ignore
from src.Tools import functions as nn_func


class Perceptron:
    def __init__(self, features, targets, learning_rate, epoche):
        self.features=features
        self.targets=targets
        self.lr=learning_rate
        self.epoche=epoche
        self.pesi=np.random.rand(features.shape[1])
        self.bias=np.random.randint(7)
        self.errori=[]


    def predict(self, feature):
        predizione=np.dot(feature, self.pesi) + self.bias
        return 1 if predizione >= 0 else 0
        

    def Errore(self, predizione, target):
        errore=predizione-target
        return errore
    

    def aggiornare_pesi(self, errore, feature):
        self.pesi -= self.lr * (errore) * feature
        self.bias -= self.lr * errore


    def Allenare(self):
        if self.features.shape > 1: 
            for epoch in range(self.epoche):
                for index, feature_batch in enumerate(self.features):
                    pred_batch=self.predict(feature=feature_batch)
                    loss_batch=self.Errore(predizione=pred_batch, target=self.targets[index])
                    self.aggiornare_pesi(errore=loss_batch, target=self.features[index])
                    loss+=loss_batch
                if epoch % 5 == 0:
                    print(f"l'epoca: {epoch}, loss: {loss}")
                    self.errori.append(loss)
        else:
            for epoch in range(self.epoche):
                pred=self.predict(feature=self.features)
                loss=self.Errore(predizione=pred)
                self.aggiornare_pesi(errore=loss, feature=self.features)
                if epoch % 5 == 0:
                    print(f"l'epoca: {epoch}, loss: {loss}")
                    self.errori.append(loss)

            
            


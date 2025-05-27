import numpy as np # type: ignore


class Perceptron:
    def __init__(self, features, targets, learning_rate, epoche):
        self.features=features
        self.targets=targets
        self.lr=learning_rate
        self.epoche=epoche
        self.pesi=np.random.randn(features.shape[1])
        self.bias=np.random.randint(7)
        self.predizioni=[]
        self.errori=[]


    def predict(self, feature):
        #print(f"feature: {feature} pesi: {self.pesi}" )
        predizione=np.dot(feature, self.pesi)
        predizione=np.sum(predizione) + self.bias
        return 1 if predizione >= 0 else 0
        

    def Errore(self, predizione, target):
        #print(f"predizione {predizione} target: {target}")
        errore=np.mean(predizione-target)
        return errore
    

    def aggiornare_pesi(self, errore, feature):
        #print(f"errore shape: {errore} feature shape: {feature} pesi shape: {self.pesi}")
        self.pesi -= self.lr * errore * feature
        self.bias -= self.lr * errore


    def Allenare(self):
        for epoch in range(self.epoche):
            for batch in range(self.features.shape[0]):
                #print(self.features[batch])
                pred=self.predict(feature=self.features[batch])
                #print(f"predizione: {pred}, target: {self.targets[batch]}")
                error=self.Errore(predizione=pred, target=self.targets[batch])
                #print(f"errore: {error}, feature: {self.features[batch]}")
                self.aggiornare_pesi(errore=error, feature=self.features[batch])
            if epoch % 5 == 0:
                print(f"epoca: {epoch} perdita: {error}")
              




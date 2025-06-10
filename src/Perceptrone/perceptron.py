import numpy as np # type: ignore


class Perceptron:
    """
    Classe principale che implementa l'archittetura e i componenti del percettrone basico

    attributi:
        self.features (np.ndarray): sono i dati di input del percettrone
        self.targets (np.ndarray): sono i dati di output, che veranno predetti
        self.lr (float): e un valore base per il learning rate
        self.epoche (int): indica quante volte il modello allenara
        self.pesi (np.ndarray): una matrice o vettore di numeri random che sono i pesi del neurone
        self.bias (float): un numero float per dire quanto biased e la predizione
        self.errori (list): una lista per mantenere traccia del errore per epoca
    """
    def __init__(self, features, targets, learning_rate, epoche):
        self.features=features
        self.targets=targets
        self.lr=learning_rate
        self.epoche=epoche
        self.pesi=np.random.randn(features.shape[1])
        self.bias=np.random.randint(7)
        self.errori=[]


    def predict(self, feature:np.ndarray):
        """
        implementa l'agoritmo di somma pesata tra i vettori/matrici d'input e pesi

        params:
            feature (np.ndarray): dati d'input
        
        returns: 
            1 se la somma attivata > zero e zero se minore
        """
        #print(f"feature: {feature} pesi: {self.pesi}" )
        predizione=np.dot(feature, self.pesi)
        predizione=np.sum(predizione) + self.bias
        return 1 if predizione >= 0 else 0
        

    def Errore(self, predizione:np.ndarray, target:np.ndarray):
        """
        calcola la differenza tra i vallori predetti e i valori target

        params: 
            predizione (np.ndarray): la predizione originata dalla predizione
            target (np.ndarray): i valori di target per il modello

        returns: 
            errore (float): uno scalare risultato della differenza
        """
        #print(f"predizione {predizione} target: {target}")
        errore=np.mean(predizione-target)
        return errore
    

    def aggiornare_pesi(self, errore:float, feature:np.ndarray):
        """
        addestra i pesi basandosi su l'errore ricavato

        params: 
            errore (float): la differenza tra le predizioni e i valori target
            feature (np.ndarray): i dati di input
        
        returns: 
            None
        """
        #print(f"errore shape: {errore} feature shape: {feature} pesi shape: {self.pesi}")
        self.pesi -= self.lr * errore * feature
        self.bias -= self.lr * errore


    def Allenare(self):
        """
        utilizza tutti le funzioni per allenare il modello nell'intervalo di n_epoche
        
        params: 
            None

        returns:
            None
        """
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
              




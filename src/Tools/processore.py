import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore


from src.Tools import functions as nn_func

class Metriche:

    def __init__(self, modello, dataset):
        self.modello=modello
        self.dataset=pd.read_csv(dataset)


    def split_data(self, fattore, features, labels):
        split=fattore*len(features)
        X_train=features[:split]
        X_test=features[:split]
        y_train=labels[split:]
        y_test=labels[split:]
        return X_train, y_train, X_test, y_test


    def cross_validation(self, K:int, features:np.ndarray, labels:np.ndarray):
        errore_training_folds=[]
        errore_testing_folds=None

        #crea un numero specifico che divide ugualmente tutti elementi dello dataset
        fold_size=len(features) // K

        #mescola i dati ogni volta che e necessario esseguire una nuova validazione
        indices=np.arange(len(features))
        np.random.shuffle(indices)
        features, labels=np.array(features[indices]), np.array(labels[indices])

        #crea una lista dove ogni elemento di essa e un'altra lista contenente fold size elementi 
        feature_folds=[features[i*fold_size:fold_size*(i+1)]for i in range(K)]
        label_folds=[labels[i*fold_size:fold_size*(i+1)]for i in range(K)]


        #allenare e testare il modello
        for i in range(K-1):
            print(f"========================================\nAlleno fold: {i}")
            X_train=feature_folds[i]
            y_train=label_folds[i]

            print(f"X_train: {X_train.shape}")
            print(f"y_train: {y_train.shape}")

            #ogni uno di essi ha una misura uguale a len(features) // K
            self.modello.features=X_train
            self.modello.targets=y_train
            self.modello.Allenare()
            errore_training_folds.append(self.modello.errori)    

        print(f"====================================\nTeste fold: {K}")
        self.modello.Allenare()
        errore_testing_folds=self.modello.errori

        
        return errore_training_folds, errore_testing_folds


class Processore:
    def __init__(self, dataset, modello):
        self.dataset=pd.read_csv(dataset)
        self.modello=modello

    #funzione che scalabilizza i dati 
    def standardizzare_data(data):
        mean=data.mean()
        std=data.std()
        standard_data=(data - mean) / std
        return standard_data

    #funzione che denormalizza i dati
    def denormalizzare_data(self, standard_data, normal_data):
        mean=normal_data.mean()
        std=normal_data.std()
        denormalized_data=standard_data * std + mean
        return denormalized_data

    #funzione che denormalizza le predizioni
    def denormalizzare_predizione(self, original_target, standard_pred):
        mean=original_target.mean()
        std=original_target.std()
        data_normalizzata=standard_pred * std + mean
        return data_normalizzata
    
    #funzione che criptofgrafa i dati categorici del datset seguendo l'algoritmo di OneHot-encoding
    def codificazione_OneHot(self, data_categorica):

        #crea un dizionario contenendo tutte le categorie per ogni valore assegnandoli 
        categorie_uniche=np.unique(data_categorica)
        categorie_indici={idx : cat for idx, cat in enumerate(categorie_uniche)}
        
        #costruisce la matrice OneHot
        OneHot=np.zeros((len(data_categorica), len(categorie_indici)), dtype=int)
        for idx, cat in enumerate(data_categorica):
            pos_OneHot=np.where(categorie_uniche == cat)[0][0]
            OneHot[idx][pos_OneHot]=1
        return categorie_indici, OneHot
            


    #funzione che decriptografa i dati categorici del dataset
    def decodificazione_OneHot(self, OneHot, categorie_indici):
        OneHot_decodificato=[]
        for cat_encoded in OneHot:
            indice=np.where(cat_encoded != 0)[0][0]
            cat_decoded=categorie_indici.get(indice)
            OneHot_decodificato.append(cat_decoded)
        return OneHot_decodificato
                
        






        
            


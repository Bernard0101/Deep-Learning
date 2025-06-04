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


    def cross_validation(self, K, features, labels, funzione_costo="MSE"):
        errore_fold=[]

        #crea un numero specifico che divide ugualmente tutti elementi dello dataset
        fold_size=len(features) // K

        #mescola i dati ogni volta che e necessario esseguire una nuova validazione
        indices=np.arange(len(features))
        np.random.shuffle(indices)
        features, labels=np.array(features[indices]), np.array(labels[indices])

        #crea una lista dove ogni elemento di essa e un'altra lista contenente fold size elementi 
        feature_folds=[features[i*fold_size:fold_size*(i+1)]for i in range(K)]
        label_folds=[labels[i*fold_size:fold_size*(i+1)]for i in range(K)]


        #allenare e testare modello
        for i in range(K-1):
            print(f"========================================\nAlleno fold: {i}")
            x_train=feature_folds[i]
            y_train=label_folds[i]

            print(f"X_train: {x_train.shape}")
            print(f"y_train: {y_train.shape}")

            self.modello.features=x_train # un array con 150 elementi
            self.modello.targets=y_train # un array con 150 elementi
            self.modello.Allenare()
            errore_fold.append(np.mean(self.modello.errori))
            
            
            print(f"media Errore Modello: {np.mean(self.modello.errori)}")
        print(f"====================================\nTeste fold: {K}")
        self.modello.Allenare()


        return errore_fold


class Processore:
    def __init__(self, dataset, modello):
        self.dataset=pd.read_csv(dataset)
        self.modello=modello

    #funzione che scalabilizza i dati 
    def standardizzare_data(dataset):
        mean=dataset.mean()
        std=dataset.std()
        standard_data=(dataset - mean) / std
        return standard_data

    #funzione che denormalizza i dati
    def denormalizzare_data(self, standard_data, colonna):
        mean=colonna.mean()
        std=colonna.std()
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

        #crea un dizionario contenendo tutte le categorie per ogni valore assegnadoli 
        categorie_uniche=np.unique(data_categorica)
        categorie_indici={cat : idx for idx, cat in enumerate(categorie_uniche)}
        
        #costruisce la matrice OneHot
        OneHot=np.zeros((len(data_categorica), len(categorie_indici)), dtype=int)
        for idx, cat in enumerate(data_categorica):
            posizione=categorie_indici[cat]
            OneHot[idx][posizione]=1
        return OneHot, categorie_indici
            
    #funzione che decriptografa i dati categorici del dataset
    def decodificazione_OneHot(self, OneHot, categorie):
        OneHot_decodificato=[]
        for cat_encoded in OneHot:
            indice=np.where(cat_encoded != 0)
            cat_decoded=categorie[indice]
            OneHot_decodificato.append(cat_decoded)
        return OneHot_decodificato
                
        






        
            


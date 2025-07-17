import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore


from src.Tools import functions as nn_func

class Metriche:
    

    def __init__(self, modello, dataset):
        self.modello=modello
        self.dataset=pd.read_csv(dataset)
        self.train_errori=[]
        self.test_errori=[]
        self.x_train=None
        self.y_train=None
        self.x_test=None
        self.y_test=None


    def split_data(self, fattore, features, labels):
        split=fattore*len(features)
        self.x_train=features[:split]
        self.x_test=features[split:]
        self.y_train=labels[:split]
        self.y_test=labels[split:]


    def cross_validation(self, K:int, features:np.ndarray, targets:np.ndarray):

        #crea un numero specifico che divide ugualmente tutti elementi dello dataset
        fold_size=len(features) // K

        #mescola i dati ogni volta che e necessario esseguire una nuova validazione
        indices=np.arange(len(features))
        np.random.shuffle(indices)
        features, targets=np.array(features[indices]), np.array(targets[indices])

        #crea una lista dove ogni elemento di essa e un'altra lista contenente fold size elementi 
        feature_folds=[features[i*fold_size:fold_size*(i+1)]for i in range(K)]
        target_folds=[targets[i*fold_size:fold_size*(i+1)]for i in range(K)]


        #allenare e testare il modello
        for i in range(K-1):
            X_train=feature_folds[i]
            y_train=target_folds[i]

            #ognuno di essi ha una misura uguale a len(features) // K
            print(f"=====================================\nAlleno: {i+1}")
            self.modello.features=X_train
            self.modello.targets=y_train
            self.modello.Allenare()
            self.train_errori.append(self.modello.errori)    

        print(f"=====================================\nAlleno: {K}")
        self.x_test=feature_folds[K-1]
        self.y_test=target_folds[K-1]
        self.modello.features=self.x_test
        self.modello.targets=self.y_test
        self.modello.Allenare()
        self.test_errori.append(self.modello.errori)


        #plottare l'errore del modello in ogni epoca per ogni fold della validazione incrociata
        totale_epoche_testing=np.arange(0, len(self.test_errori[0]), 1)
        fig, asse=plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
        for i, ax in enumerate(asse.flatten()):
            limite=min(len(self.train_errori[i]), len(self.test_errori[0]))
            totale_epoche=np.arange(0, limite, 1)
            ax.plot(totale_epoche, self.train_errori[i][:limite], lw=5, c="red", label=f"dati di allenamento fold: {i+1}")
            ax.plot(totale_epoche, self.test_errori[0][:limite], lw=5, c="cornflowerblue", label=f"dati di valutazione fold: {K}")
            ax.set_title(f"Errore folder {i+1}")
            ax.set_xlabel("Iterazioni (epoche)")
            ax.set_ylabel(f"funzione costo: {self.modello.loss_fn}")
            ax.grid(True)
            ax.legend()
        plt.suptitle(f"Analisi Progresso Allenamento del modello, ottimizzattore: {self.modello.optim}")
        plt.show()


        #plottare l'errore del modello nella fase di valutazione con l'ultimo fold della validazione incrociata
       
    

    def curva_apprendimento(self):
        pass

class processore:
    def __init__(self, dataset, modello):
        self.dataset=pd.read_csv(dataset)
        self.modello=modello

    #scalabilizzatore 
    def standard_scaler(data):
        data=np.array(data)
        mean=np.mean(data)
        std=np.std(data)
        scaled_data=(data - mean) / std
        return scaled_data
    

    def minMax_scaler(data):
        data=np.array(data)
        min_data=np.min(data)
        max_data=np.max(data)
        scaled_data=(data - min_data)/(max_data - min_data)
        return scaled_data
    

    def Robust_scaler(data):
        data=np.array(data)
        sorted_data=np.sort(data)
        Q1=np.percentile(sorted_data, 25)
        Q2=np.percentile(sorted_data, 50)
        Q3=np.percentile(sorted_data, 75)
        scaled_data=(data - Q2)/(Q3 - Q1)
        return scaled_data
    
    def log_scaler(data):
        data=np.array(data)
        eps=1e-12
        scaled_data=np.log10(data + eps)
        return scaled_data

    

    
    
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
                
        






        
            


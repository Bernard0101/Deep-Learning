import PIL
import PIL.Image
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore

from src.Rete_Neurale_Multistrato import NeuralNetwork as nn
from src.Tools import utils
from src.Tools import functions as nn_func
from src.Tools import PIML


#scaricare il dataset e trasformarlo in un dataframe e fare feature engineering
data_path="Datasets/Legge_di_Coulomb.csv"
df=pd.read_csv(data_path)


#trasformare i dati di uC e pC, nel rispettivo valore numerico
df["q1_unita"]=df["q1_unita"].replace({'uC' : 1e-6, 'pC' : 1e-12})
df["q2_unita"]=df["q2_unita"].replace({'uC' : 1e-6, 'pC' : 1e-12})

#passare i valori per un formato numerico
cariche_1=df["Carica_1 (C)"].values
cariche_2=df["Carica_2 (C)"].values
q1_unit=df["q1_unita"].values
q2_unit=df["q2_unita"].values

#creare le due nuove colonne che saranno utilizzate come features
carica_q1=(df["Carica_1 (C)"].values * df["q1_unita"].values)
carica_q2=(df["Carica_2 (C)"].values * df["q2_unita"].values)
df["Carica_1 (C)"]=carica_q1
df["Carica_2 (C)"]=carica_q2
del df["q1_unita"]
del df["q2_unita"]
print(df.head())


#verifica della coerenza del dataset con la legge fisica
forza_elettrica=PIML.Fisica.Forza_elettrica_leggeCoulomb(q1=df["Carica_1 (C)"].values, q2=df["Carica_2 (C)"].values, dist=df["distanza (m)"].values)
loss=nn_func.Perdita.Loss_MAE(self=nn_func.Perdita, y_pred=df["forza (N)"].values, y_label=forza_elettrica)
print(loss)


#feature engieneering con polynomial features per rendere i legami tra i dati piu facili da capirsi
carica_q1_norm=np.absolute(df["Carica_1 (C)"].values - min(df["Carica_1 (C)"].values) / max(df["Carica_1 (C)"].values) - min(df["Carica_1 (C)"].values))
carica_q2_norm=np.absolute(df["Carica_2 (C)"].values - min(df["Carica_2 (C)"].values) / max(df["Carica_2 (C)"].values) - min(df["Carica_2 (C)"].values))
distanza=df["distanza (m)"].values
prodotto_cariche=np.absolute(df["Carica_1 (C)"].values * df["Carica_2 (C)"].values)
rapporto_distanza=1 / df["distanza (m)"].values
quadrato_distanza=df["distanza (m)"].values ** 2
rapporto_quadrato_distanza=1 / df["distanza (m)"].values ** 2
legge_coulomb=np.absolute(carica_q1 * carica_q2) / np.power(distanza, 2)

#definizione dei targets come la il log della norma della forza 
F_log_norm=np.log10(np.absolute(df["forza (N)"].values) / np.absolute(prodotto_cariche))


#separazione del dataset in features e labels
data_features=np.column_stack((prodotto_cariche, quadrato_distanza))
data_targets=F_log_norm

#plottare i dati puri rumurosi
plt.figure(figsize=(12, 8))
plt.scatter(df["distanza (m)"].values, y=df["forza (N)"].values, alpha=0.7, c="mediumblue", label="dati puri rumurosi")
plt.title("Analisi del contenuto del dataset")
plt.xlabel("distanza (m^2)")
plt.ylabel("Forza (N)")
plt.grid(True)
plt.show()


#plottare la distanza contro la forza elettrica e il valore del prodotto tra le cariche
plt.figure(figsize=(12, 8))
sc=plt.scatter(df["distanza (m)"].values, df["forza (N)"].values, c=np.log10(np.abs(prodotto_cariche)), cmap="plasma", s=40, alpha=0.7)
plt.colorbar(sc, label='prodotto assulto tra le cariche legge: |q1 * q2|')
plt.yscale("log")
plt.title("Analisi della distribuizione della potenza delle cariche")
plt.xlabel("distanza (m^2)")
plt.ylabel("Forza normalizzata (N / C²))")
plt.grid(True)
plt.show()


#plottare la relazione 1/r^2 che la rete dovra apprendere
plt.figure(figsize=(12, 8))
plt.plot(quadrato_distanza, F_log_norm, c="orange", alpha=0.7, label="F~=1/r^2")
plt.yscale("log")
plt.title("Legame tra la distanza e la forza elettrica compiuta tra le cariche")
plt.xlabel("rapporto quadrato della distanza (1/r^2)")
plt.ylabel("logaritmo della norma della forza")
plt.grid(True)
plt.legend()
plt.show()


#instanziare il modello di rete neurale, con tutti i parametri
NeuralNet=nn.Architettura(nn_layers=[2, 8, 16, 16, 8, 1], init_pesi="He", epochs=500,
                                        features=data_features, targets=data_targets, learning_rate=5e-3, 
                                        ottimizzattore="Adagrad", funzione_perdita="MSE", attivazione="leaky_ReLU")

#istanziare l'oggetti per organizzazione e processamento dei dati
metriche=utils.Metriche(dataset=data_path, modello=NeuralNet)

#allenare e valutare il modello con la validazione incrociata
K_folds=5
metriche.cross_validation(K=K_folds, features=data_features, targets=data_targets)
forza_elettrica=PIML.Fisica.Forza_elettrica_leggeCoulomb(q1=df["Carica_1 (C)"].values, q2=df["Carica_2 (C)"].values, dist=df["distanza (m)"].values)



predizioni=NeuralNet.predict(inputs=metriche.x_test)
plt.figure(figsize=(10, 6))
plt.scatter(metriche.x_test[:, 1], metriche.y_test, s=25, c="mediumblue", label=f"dati dataset")
plt.scatter(metriche.x_test[:, 1], predizioni, s=25, c="orangered", label=f"predizioni modello")
plt.title("Errore Test Fold")
plt.xlabel("distanza in metri")
plt.ylabel("forza in newtons")
plt.grid(True)
plt.legend()
plt.show()


n_folds=np.arange(0, len(metriche.test_errori), 1)
media_errore_folds=[np.mean(i) for i in metriche.test_errori]
plt.figure(figsize=(12, 8))
plt.bar(n_folds, media_errore_folds, color="navy", label="Errore per fold")
plt.title("Analise Validazione-incrociata")
plt.xlabel("K-folds")
plt.ylabel("Errore complessivo")
plt.grid(True)
plt.legend()
plt.show()

#elaborazione
predizioni_log=NeuralNet.predict(inputs=data_features)
predizioni=np.power(10, predizioni_log)
segni=np.sign(forza_elettrica)
predizioni=predizioni.flatten() * segni.flatten()
Forza=df["forza (N)"].values / np.absolute(prodotto_cariche)

print(f"preds: {predizioni.shape} Forza: {Forza.shape}")
plt.figure(figsize=(12, 8))
plt.scatter(df["distanza (m)"].values, Forza, c="mediumblue", alpha=0.4, label="dati rumurosi")
plt.scatter(df["distanza (m)"].values, predizioni, alpha=0.7, c="limegreen", label="predizioni del modello")
plt.title("Analise Prestazione modello")
plt.xlabel("Distanza (m^2)")
plt.ylabel("F_norm=Forza / |q1 * q2|")
plt.grid(True)
plt.legend()
plt.show()

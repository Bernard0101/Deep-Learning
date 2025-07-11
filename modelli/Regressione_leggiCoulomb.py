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
df["carica_q1"]=(cariche_1 * q1_unit)
df["carica_q2"]=(cariche_2 * q2_unit)
print(df.head())

#verifica della coerenza del dataset con la legge fisica
forza_elettrica=PIML.Fisica.Forza_elettrica_leggeCoulomb(q1=df["carica_q1"].values, q2=df["carica_q2"].values, dist=df["distanza (m)"].values)
loss=nn_func.Perdita.Loss_MAE(self=nn_func.Perdita, y_pred=df["forza (N)"].values, y_label=forza_elettrica)
print(loss)

#carica_q1=utils.processore.standardizzare_data(data=df["carica_q1"].values)
#carica_q2=utils.processore.standardizzare_data(data=df["carica_q2"].values)
#distanza=utils.processore.standardizzare_data(data=df["distanza (m)"].values)
#forza=utils.processore.standardizzare_data(data=df["forza (N)"].values)

#feature engieneering per rendere i legami tra i dati piu facili da capirsi
prodotto_cariche=df["carica_q2"].values * df["carica_q1"].values
distanza=df["distanza (m)"].values
rapporto_distanza=1 / df["distanza (m)"].values
quadrato_distanza=df["distanza (m)"] ** 2

#separazione del dataset in features e labels
data_features=np.column_stack((prodotto_cariche, distanza, rapporto_distanza, quadrato_distanza))
data_targets=df["forza (N)"].values
K_folds=6

#assicurare le dimensioni dei features e targets
print(f"features: {data_features.shape}")
print(f"targets: {data_targets.shape}")


#instanziare il modello di rete neurale, con tutti i parametri
NeuralNet=nn.Architettura(nn_layers=[4, 4, 4, 1], init_pesi="He", epochs=1000,
                                        features=data_features, targets=data_targets, learning_rate=5e-3, 
                                        ottimizzattore="Adagrad", funzione_perdita="MSE", attivazione="Tanh")

#istanziare l'oggetti per organizzazione e processamento dei dati
metriche=utils.Metriche(dataset=data_path, modello=NeuralNet)
processore_dati=utils.processore(dataset=data_path, modello=NeuralNet)

#allenare e valutare il modello con la validazione incrociata
errore_training_folds, errore_testing_folds=metriche.cross_validation(K=K_folds, features=data_features, targets=data_targets)
forza_elettrica=PIML.Fisica.Forza_elettrica_leggeCoulomb(q1=df["carica_q1"].values, q2=df["carica_q2"].values, dist=df["distanza (m)"].values)


#plottare la distanza contro la forza elettrica e il valore del prodotto tra le cariche
F_norm=forza_elettrica / np.absolute(prodotto_cariche)
plt.figure(figsize=(12, 8))
sc=plt.scatter(df["distanza (m)"].values, F_norm, c=np.log10(np.abs(prodotto_cariche)), cmap="plasma", s=40, alpha=0.7)
plt.colorbar(sc, label='prodotto assulto tra le cariche legge: |q1 * q2|')
plt.yscale("log")
plt.title("Legame: Distanza contro Forza tra le cariche")
plt.xlabel("distanza (m^2)")
plt.ylabel("Forza normalizzata (N / C²))")
plt.legend()
plt.grid(True)
plt.show()


#plottare l'errore del modello in ogni epoca per ogni fold della validazione incrociata
fig, asse=plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
for i, ax in enumerate(asse.flatten()):
    ax.plot(np.arange(0, NeuralNet.epoche[i]+1, 1), errore_training_folds[i], c="red", label=f"Errore folder {i+1}")
    ax.set_title(f"Errore folder {i+1}")
    ax.set_xlabel("Iterazioni (epoche)")
    ax.set_ylabel("Sbaglio")
    ax.grid(True)
    ax.legend()
fig.suptitle(f"Analisi Progresso Allenamento del modello, ottimizzattore: {NeuralNet.optim}")
plt.show()


#plt.figure(figsize=(10, 6))
#lt.scatter(metriche.x_test[:, 1], metriche.y_test, s=25, c="mediumblue", label=f"dati dataset")
#plt.scatter(metriche.x_test[:, 1], NeuralNet.preds[:, 0], s=25, c="orangered", label=f"predizioni modello")
#plt.title("Errore Test Fold")
#plt.xlabel("distanza in metri")
#plt.ylabel("forza in newtons")
#plt.grid(True)
#plt.legend()
#plt.show()

media_errore_folds=[np.mean(i) for i in errore_training_folds]
plt.figure(figsize=(12, 8))
plt.bar(np.arange(0, len(errore_training_folds), 1), media_errore_folds, color="navy", label="Errore per fold")
plt.title("Analise Validazione-incrociata")
plt.xlabel("K-folds")
plt.ylabel("Errore complessivo")
plt.grid(True)
plt.legend()
plt.show()


predizioni=NeuralNet.predict(inputs=data_features)
plt.figure(figsize=(12, 8))
plt.scatter(df["distanza (m)"].values, df["forza (N)"].values, c="mediumblue", alpha=0.4, label="dati rumurosi")
plt.scatter(df["distanza (m)"].values, predizioni, alpha=0.7, c="limegreen", label="predizioni del modello")
plt.xscale("log")
plt.yscale("log")
plt.title("Analise Prestazione modello")
plt.xlabel("Distanza in metri")
plt.ylabel("Forza in newtons")
plt.grid(True)
plt.legend()
plt.show()
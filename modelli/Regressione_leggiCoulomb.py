import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore


from src.Rete_Neurale_Multistrato import NeuralNetwork as nn
from src.Tools import processore 
from src.Tools import functions as nn_func
from src.Tools import PIML

data_path="Datasets/fisica_Legge_Coulomb.csv"
data=pd.read_csv(data_path)

#feature engineering per rendere le features piu adatte all'apprendimento della rete
data_std=processore.Processore.standardizzare_data(data=data)

#separazione del dataset in features e labels
print(data_std.head())
data_features=data_std[["Carga 1 (Coulombs)", "Carga 2 (Coulombs)", "Distanza (m)"]].values
data_targets=data_std["Forza (N)"].values
K_folds=5


#alleno del modello, e utilizzazione di metriche per valutazione
NeuralNet=nn.Architettura(nn_layers=[3, 8, 8, 8, 1], init_pesi="Xavier", epochs=1000,
                                        features=data_features, targets=data_targets, learning_rate=3e-3, 
                                        ottimizzattore="SGD", funzione_perdita="MSE", attivazione="leaky_ReLU")

print(len(data_features))
print(data_features.shape[0])
print(NeuralNet.features.shape[0])

processore_dati=processore.Metriche(dataset=data_path, modello=NeuralNet)


errore_training_folds, errore_testing_folds=processore_dati.cross_validation(K=K_folds, features=data_features, labels=data_targets)
pred=NeuralNet.predict(inputs=data_features)
predizione_denormalizzate=processore.Processore.denormalizzare_predizione(processore, original_target=data_targets, standard_pred=pred)
features=data[["Carga 1 (Coulombs)", "Carga 2 (Coulombs)", "Distanza (m)"]].values
forza_elettrica=PIML.Fisica.Forza_elettrica_leggeCoulomb(features=features)

plt.figure(figsize=(12, 8))
plt.scatter(x=data["Distanza (m)"].values, y=data["Forza (N)"].values, c="mediumblue", alpha=0.6, label="dataset puro rumuroso")
plt.scatter(x=data["Distanza (m)"].values, y=forza_elettrica, c="orangered", alpha=0.6, label="Dataset basato Legge fisica")
plt.yscale("log")
plt.title("Comparazione Forza e Distanza legge di Ohm")
plt.xlabel("Distanza (m²)")
plt.ylabel("Forza (N)")
plt.grid(True)
plt.legend()
plt.show()



fig, asse=plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
for i, ax in enumerate(asse.flatten()):
    ax.plot(np.arange(0, NeuralNet.epoche[i]+1, 1), errore_training_folds[i], c="red", label=f"Errore folder {i+1}")
    ax.set_title(f"Errore folder {i+1}")
    ax.set_xlabel("Iterazioni (epoche)")
    ax.set_ylabel("Sbaglio")
    ax.grid(True)
    ax.legend()
fig.suptitle("Analisi Progresso Allenamento del modello")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(np.arange(0, NeuralNet.epoche[-1]+1, 1), errore_testing_folds, c="red")
plt.title("Analisi Errore Testing phase")
plt.xlabel("iterazioni (Epoche)")
plt.ylabel("Sbaglio")
plt.grid(True)
plt.show()

media_errore_folds=[np.mean(i) for i in errore_training_folds]

plt.figure(figsize=(12, 8))
plt.bar(np.arange(0, len(errore_training_folds), 1), media_errore_folds, color="coral", label="Errore per fold")
plt.title("Analise Validazione-incrociata")
plt.xlabel("K-folds")
plt.ylabel("Errore complessivo")
plt.grid(True)
plt.legend()
plt.show()


plt.figure(figsize=(12, 8))
plt.scatter(data["Distanza (m)"].values, data["Forza (N)"].values, c="mediumblue", alpha=0.4, label="dati reali rumurosi")
plt.scatter(data["Distanza (m)"].values, forza_elettrica, c="darkorange", alpha=0.6, label="dati basati sulla legge fisica")
plt.scatter(data["Distanza (m)"].values, pred, alpha=0.7, c="limegreen", label="predizione modello")
plt.yscale("log")
plt.title("Analise Prestazione modello")
plt.xlabel("Distanza in metri")
plt.ylabel("Forza in newtons")
plt.grid(True)
plt.legend()
plt.show()
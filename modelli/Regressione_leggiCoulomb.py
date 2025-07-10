import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore

from src.Rete_Neurale_Multistrato import NeuralNetwork as nn
from src.Tools import processore 
from src.Tools import functions as nn_func
from src.Tools import PIML

data_path="Datasets/fisica_Legge_Coulomb.csv"
dataset=pd.read_csv(data_path)
data_std=processore.Processore.standardizzare_data(data=dataset)

#feature engineering per rendere le features piu adatte all'apprendimento della rete
#data_std=processore.Processore.standardizzare_data(data=data)

#separazione del dataset in features e labels
print(data_std.head())
data_features=data_std[["Carga 1 (Coulombs)", "Carga 2 (Coulombs)", "Distanza (m)"]].values
data_targets=data_std["Forza (N)"].values
K_folds=6


#alleno del modello, e utilizzazione di metriche per valutazione
NeuralNet=nn.Architettura(nn_layers=[3, 6, 4, 1], init_pesi="Xavier", epochs=1000,
                                        features=data_features, targets=data_targets, learning_rate=3e-2, 
                                        ottimizzattore="Adagrad", funzione_perdita="MSE", attivazione="leaky_ReLU")


metriche=processore.Metriche(dataset=data_path, modello=NeuralNet)
processore_dati=processore.Processore(dataset=data_path, modello=NeuralNet)


errore_training_folds, errore_testing_folds=metriche.cross_validation(K=K_folds, features=data_features, targets=data_targets)
forza_elettrica=PIML.Fisica.Forza_elettrica_leggeCoulomb(q1=dataset["Carga 1 (Coulombs)"].values, q2=dataset["Carga 2 (Coulombs)"].values, dist=dataset["Distanza (m)"].values)

plt.figure(figsize=(12, 8))
plt.scatter(x=dataset["Distanza (m)"].values, y=dataset["Forza (N)"].values, c="mediumblue", alpha=0.6, label="dataset puro rumuroso")
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
fig.suptitle(f"Analisi Progresso Allenamento del modello, ottimizzattore: {NeuralNet.optim}")
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(metriche.x_test[:, 2], metriche.y_test, s=25, c="mediumblue", label=f"dati dataset")
plt.scatter(metriche.x_test[:, 2], NeuralNet.preds[:, 0], s=25, c="orangered", label=f"predizioni modello")
plt.title("Errore Test Fold")
plt.xlabel("distanza in metri")
plt.ylabel("forza in newtons")
plt.grid(True)
plt.legend()
plt.show()

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
plt.scatter(dataset["Distanza (m)"].values, dataset["Forza (N)"].values, c="mediumblue", alpha=0.4, label="dati rumurosi")
plt.scatter(dataset["Distanza (m)"].values, predizioni, alpha=0.7, c="limegreen", label="predizioni del modello")
plt.yscale("log")
plt.title("Analise Prestazione modello")
plt.xlabel("Distanza in metri")
plt.ylabel("Forza in newtons")
plt.grid(True)
plt.legend()
plt.show()
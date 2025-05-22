import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore


from Rete_Neurale_Multistrato import NeuralNetwork
from Tools import functions
from Tools import processore 

#separazione in del dataset in features e labels
data_path="Datasets/Dataset_Legge_Di_Coulomb.csv"
data=pd.read_csv(data_path)
data_std=processore.Processore.standartizzareData(dataset=data)
data_features=data_std[["Carga 1 (Coulombs)","Carga 2 (Coulombs)","Distanza (m)"]].values
data_targets=data_std["Forza (N)"].values
K_folds=5

#alleno del modello, e utilizzazione di metriche per valutazione
NeuralNet=NeuralNetwork.nn_Architettura(nn_layers=[9, 6, 3, 1], inputs=3, init_pesi="He", features=data_features, targets=data_targets, epoche=25, learning_rate=0.003)
processore_dati=processore.Metriche(dataset=data_path, modello=NeuralNet)
errore_fold, ordine=processore_dati.cross_validation(K=K_folds, features=data_features, labels=data_targets, funzione_costo="MSE")

predizione_denormalizzate=processore.Processore.denormalizzarePredizione(processore, target=data_targets, dataset=data)


plt.figure(figsize=(12, 8))
plt.scatter(x=data["Distanza (m)"].values, y=data["Forza (N)"].values, c="darkorange", alpha=0.03)
plt.title("Comparazione Forza e Distanza legge di Ohm")
plt.xlabel("Distanza (m²)")
plt.ylabel("Forza (N)")
plt.yscale("log")
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(NeuralNet.errori, np.arange(0, len(NeuralNet.errori), 1), c="r", label="errore per epoca")
plt.title("Progresso complessivo del modello")
plt.xlabel("Iterazioni (epoche)")
plt.ylabel("Sbaglio")
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
plt.bar()
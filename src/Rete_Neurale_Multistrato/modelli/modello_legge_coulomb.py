import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore


from src.Rete_Neurale_Multistrato import NeuralNetwork
from src.Tools import processore 

#separazione in del dataset in features e labels
data_path="Datasets/Dataset_Legge_Di_Coulomb.csv"
data=pd.read_csv(data_path)
distanza_M=data["Distanza (m)"].values
forza_N=data["Forza (N)"].values

data_std=processore.Processore.standardizzare_data(dataset=data)
data_features=data_std[["Carga 1 (Coulombs)","Carga 2 (Coulombs)","Distanza (m)"]].values
data_targets=data_std["Forza (N)"].values
K_folds=5

#alleno del modello, e utilizzazione di metriche per valutazione
NeuralNet=NeuralNetwork.nn_Architettura(nn_layers=[9, 6, 3, 1], inputs=3, init_pesi="He", features=data_features, targets=data_targets, epoche=25, learning_rate=0.003)
processore_dati=processore.Metriche(dataset=data_path, modello=NeuralNet)
errore_folds, ordine=processore_dati.cross_validation(K=K_folds, features=data_features, labels=data_targets, funzione_costo="MSE")

predizione_denormalizzate=processore.Processore.denormalizzare_predizione(processore, target=data_targets, dataset=data)

print(predizione_denormalizzate.shape)
print(distanza_M.shape)
print(forza_N.shape)

plt.figure(figsize=(12, 8))
plt.scatter(x=distanza_M, y=forza_N, c="darkorange", alpha=0.03)
plt.title("Comparazione Forza e Distanza legge di Ohm")
plt.xlabel("Distanza (m²)")
plt.ylabel("Forza (N)")
plt.yscale("log")
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(NeuralNet.errori, np.arange(0, len(NeuralNet.errori), 1), c="red", label="Errore alleno")
plt.title("Progresso complessivo del modello")
plt.xlabel("Iterazioni (epoche)")
plt.ylabel("Sbaglio")
plt.grid(True)
plt.show()


plt.figure(figsize=(12, 8))
plt.bar(np.arange(0, len(errore_folds), 1), errore_folds, color="deepskyblue", label="Errore per fold")
plt.title("Analise Validazione-incrociata")
plt.xlabel("K-folds")
plt.ylabel("Errore complessivo")
plt.grid(True)
plt.show()


plt.figure(figsize=(12, 8))
plt.scatter(x=distanza_M, y=forza_N, c="darkorange", alpha=0.03, label="Target")
plt.plot(np.arange(0, len(predizione_denormalizzate), 1), predizione_denormalizzate, c="cyan", label="Predizioni")
plt.title("Analise Regressione")
plt.grid(True)
plt.show()
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore


from src.Rete_Neurale_Multistrato import NeuralNetwork
from src.Tools import processore 
from src.Tools import functions as nn_func

data_path="Datasets/fisica_Legge_Coulomb.csv"
data=pd.read_csv(data_path)

#feature engineering per rendere le features piu adatte all'apprendimento della rete
data["Termine_fisico"]=abs(data["Carga 1 (Coulombs)"] * data["Carga 2 (Coulombs)"] / data["Distanza (m)"] ** 2)

termine_fisico=data["Termine_fisico"].values
forza=data["Forza (N)"].values

data["log_Termine_fisico"]=np.log10(data["Termine_fisico"].values + 1e-12)
data["log_Forza (N)"]=np.log10(data["Forza (N)"].values + 1e-6)

data_std=processore.Processore.standardizzare_data(dataset=data)

#separazione del dataset in features e labels
data_features=data_std[["log_Termine_fisico"]].values
data_targets=data_std["log_Forza (N)"].values
K_folds=5

print(data.head())

#alleno del modello, e utilizzazione di metriche per valutazione
NeuralNet=NeuralNetwork.nn_Architettura(nn_layers=[1, 8, 6, 8, 1], init_pesi="Xavier", features=data_features, targets=data_targets, 
                                        epoche=300, learning_rate=0.006, ottimizzattore="SGD", funzione_perdita="MSE")


processore_dati=processore.Metriche(dataset=data_path, modello=NeuralNet)


errore_folds=processore_dati.cross_validation(K=K_folds, features=data_features, labels=data_targets, funzione_costo="MSE")
pred=NeuralNet.predict(features=data_features)
predizione_denormalizzate=processore.Processore.denormalizzare_predizione(processore, original_target=data_targets, standard_pred=pred)
print(f"perdita MAE: {nn_func.nn_functions.Loss_MAE(y_pred=predizione_denormalizzate, y_label=data['Forza (N)'].values)}")


plt.figure(figsize=(10, 6))
plt.scatter(data["Termine_fisico"].values, data["Forza (N)"].values)
plt.xlabel("Termine Fisico")
plt.ylabel("Forza (N)")
plt.title("Distribuzione dei dati")
plt.show()


plt.figure(figsize=(12, 8))
plt.scatter(x=data["Distanza (m)"].values, y=data["Forza (N)"].values, c="darkorange", alpha=0.6)
plt.yscale("log")
plt.title("Comparazione Forza e Distanza legge di Ohm")
plt.xlabel("Distanza (m²)")
plt.ylabel("Forza (N)")
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(NeuralNet.errori, np.arange(0, NeuralNet.epoche, 1), c="red", label="Errore alleno")
plt.title("Progresso complessivo del modello")
plt.xlabel("Iterazioni (epoche)")
plt.ylabel("Sbaglio")
plt.grid(True)
plt.legend()
plt.show()


plt.figure(figsize=(12, 8))
plt.bar(np.arange(0, len(errore_folds), 1), errore_folds, color="deepskyblue", label="Errore per fold")
plt.title("Analise Validazione-incrociata")
plt.xlabel("K-folds")
plt.ylabel("Errore complessivo")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
plt.scatter(x=data["Distanza (m)"].values, y=data["Forza (N)"].values, c="darkorange", alpha=0.6, label="dati reali rumurosi")
plt.scatter(x=data["Distanza (m)"].values, y=predizione_denormalizzate, c="limegreen", alpha=0.6, label="predizione modello")
plt.yscale("log")
plt.title("Analise Prestazione modello")
plt.xlabel("Distanza in metri")
plt.ylabel("Forza in newtons")
plt.grid(True)
plt.legend()
plt.show()
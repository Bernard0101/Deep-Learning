import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore


from src.Rete_Neurale_Multistrato import NeuralNetwork
from src.Tools import processore 
from src.Tools import functions as nn_func

data_path="Datasets/fisica_Legge_Coulomb.csv"
data=pd.read_csv(data_path)

#feature engineering per rendere le features piu adatte all'apprendimento della rete
data_std=processore.Processore.standardizzare_data(dataset=data)

#separazione del dataset in features e labels
print(data_std.head())
data_features=data_std[["Carga 1 (Coulombs)", "Carga 2 (Coulombs)", "Distanza (m)"]].values
data_targets=data_std["Forza (N)"].values
K_folds=5


#alleno del modello, e utilizzazione di metriche per valutazione
NeuralNet=NeuralNetwork.nn_Architettura(nn_layers=[3, 8, 6, 8, 1], init_pesi="He", epoche=100,
                                        features=data_features, targets=data_targets, learning_rate=0.0003, 
                                        ottimizzattore="SGD", funzione_perdita="MSE", attivazione="leaky_ReLU")


processore_dati=processore.Metriche(dataset=data_path, modello=NeuralNet)


errore_folds=processore_dati.cross_validation(K=K_folds, features=data_features, labels=data_targets)
pred=NeuralNet.predict(features=data_features)
predizione_denormalizzate=processore.Processore.denormalizzare_predizione(processore, original_target=data_targets, standard_pred=pred)
print(f"perdita MAE: {nn_func.nn_functions.Loss_MAE(y_pred=predizione_denormalizzate, y_label=data['Forza (N)'].values)}")


plt.figure(figsize=(12, 8))
plt.scatter(x=data["Distanza (m)"].values, y=data["Forza (N)"].values, c="darkorange", alpha=0.6)
plt.yscale("log")
plt.title("Comparazione Forza e Distanza legge di Ohm")
plt.xlabel("Distanza (m²)")
plt.ylabel("Forza (N)")
plt.grid(True)
plt.show()


fig, asse=plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
for i, ax in enumerate(asse.flatten()):
    ax.plot(np.arange(0, NeuralNet.epoche, 1), errore_folds[i], c="red", label=f"Errore folder {i+1}")
    ax.set_title(f"Errore folder {i+1}")
    ax.set_xlabel("Iterazioni (epoche)")
    ax.set_ylabel("Sbaglio")
    ax.grid(True)
    ax.legend()
fig.suptitle("Analisi Progresso complessivo del modello")
plt.show()

media_errore_folds=[np.mean(i) for i in errore_folds]

plt.figure(figsize=(12, 8))
plt.bar(np.arange(0, len(errore_folds), 1), media_errore_folds, color="coral", label="Errore per fold")
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
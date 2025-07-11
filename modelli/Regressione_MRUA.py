import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore

from src.Rete_Neurale_Multistrato import NeuralNetwork as nn
from Tools import utils
from src.Tools import functions as nn_func

file_path="Datasets/fisica_MRUA.csv"
data=pd.read_csv(file_path)

data_std=utils.Processore.standardizzare_data(data=data)

print(data.head())
features=data_std[["Tempo (s)","Accelerazione (m/s²)","Velocita (m/s)"]].values
targets=data_std["Distanza (m)"].values

NeuralNet=nn.Architettura(features=features, targets=targets, nn_layers=[3, 4, 4, 4, 1], 
                                        init_pesi="He", attivazione="leaky_ReLU", funzione_perdita="MSE", 
                                        ottimizzattore="SGD", learning_rate=0.001, epochs=50)

Processore=utils.Metriche(modello=NeuralNet, dataset=file_path)

k_folds=5
errore_training_folds, errore_testing_folds=Processore.cross_validation(K=k_folds, features=NeuralNet.features, labels=NeuralNet.targets)

pred=NeuralNet.predict(features=features)
#print(f"Perdita MAE: {nn_func.nn_functions.Loss(y_target=targets, y_pred=pred, type="MAE", derivata=False)}")


plt.figure(figsize=(10, 6))
plt.plot(data["Distanza (m)"].values, data["Velocita (m/s)"], color="mediumturquoise", lw=4)
plt.title("Relazione Distanza (m) per Velocita (m/s)")
plt.xlabel("Distanza (m)")
plt.ylabel("Velocita (m/s)")
plt.grid(True)
plt.show()

fig, asse=plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
for i, ax in enumerate(asse.flatten()):
    ax.plot(np.arange(0, NeuralNet.epoche[i]+1, 1), errore_training_folds[i], color="red")
    ax.set_title(f"Errore folder {i+1}")
    ax.set_xlabel("Epoche")
    ax.set_ylabel("sbaglio")
    ax.grid(True)
fig.suptitle("Analise apprendimento complessivo modello")
plt.show()

plt.figure(figsize=(12, 8))
plt.bar(x=np.arange(0, len(errore_training_folds), 1), height=[np.mean(i) for i in errore_training_folds], color="darkslateblue", label="Errore per fold")
plt.title("Analise validazione incrociata")
plt.xlabel("K_folds")
plt.ylabel("sbaglio")
plt.grid(True)
plt.show()


plt.figure(figsize=(12, 8))
plt.plot(data["Distanza (m)"].values, data["Velocita (m/s)"], color="mediumturquoise", lw=4, label="dati reali")
plt.scatter(pred, data["Velocita (m/s)"].values, color="orangered", lw=3, label="predizioni modello")
plt.title("Analisi Prestazione Modello Compito di Regressione")
plt.xlabel("distanza (m)")
plt.ylabel("velocita (m/s)")
plt.grid(True)
plt.show()





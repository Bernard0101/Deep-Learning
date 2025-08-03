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
df["q1_unita"]=df["q1_unita"].replace({'uC' : 1e-6, 'nC' : 1e-9})
df["q2_unita"]=df["q2_unita"].replace({'uC' : 1e-6, 'nC' : 1e-9})

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

#applicare undersamppling ai dati della forza piu vicini a zero
forze_bilanciate=(df["forza (N)"] <= -6e-2) | (df["forza (N)"] >= 6e-2)
df_bilanciato=df[forze_bilanciate]
print(df_bilanciato.head())

#verifica della coerenza del dataset con la legge fisica
forza_elettrica=PIML.Fisica.Forza_elettrica_leggeCoulomb(q1=df_bilanciato["Carica_1 (C)"].values, q2=df_bilanciato["Carica_2 (C)"].values, dist=df_bilanciato["distanza (m)"].values)
loss=nn_func.Perdita.Loss_MAE(self=nn_func.Perdita, y_pred=df_bilanciato["forza (N)"].values, y_label=forza_elettrica)

plt.figure(figsize=(12, 8))
plt.scatter(df_bilanciato["distanza (m)"].values, df_bilanciato["forza (N)"].values, alpha=0.6, color="mediumblue", label="dati puri rumurosi")
plt.scatter(df_bilanciato["distanza (m)"].values, forza_elettrica, color="orange", alpha=0.6, label="legge di Coulomb")
plt.title("Verifica se i dati seguono la legge fisica")
plt.xlabel("distanza (m)")
plt.ylabel("Forza (N)")
plt.legend()
plt.grid(True)
plt.show()

#feature engieneering con polynomial features per rendere i legami tra i dati piu facili da capirsi
prodotto_cariche=df_bilanciato["Carica_1 (C)"].values * df_bilanciato["Carica_2 (C)"].values
quadrato_distanza=df_bilanciato["distanza (m)"].values ** 2
rapporto_distanza=1 / df_bilanciato["distanza (m)"].values ** 2
forza_elettrica=df_bilanciato["forza (N)"].values

#scalabilizzare i dati con il metodo robust scaling
segni=np.sign(prodotto_cariche)
scaled_prodotto_cariche=utils.processore.log_scaler(data=np.absolute(prodotto_cariche))
scaled_quadrato_distanza=utils.processore.log_scaler(data=quadrato_distanza)
scaled_rapporto_distanza=utils.processore.log_scaler(data=rapporto_distanza)
scaled_forza_elettrica=utils.processore.log_scaler(data=np.absolute(forza_elettrica))

#separazione del dataset in features e labels
data_features=np.column_stack((scaled_quadrato_distanza, scaled_prodotto_cariche))
data_targets=scaled_forza_elettrica

#plottare la distanza contro la forza elettrica e la variazione del prodotto tra le cariche
plt.figure(figsize=(16, 12))
plt.subplot(211)
sc=plt.scatter(df_bilanciato["distanza (m)"].values, df_bilanciato["forza (N)"].values, c=np.log10(prodotto_cariche), cmap="plasma", s=40)
plt.yscale("log")
plt.colorbar(sc, label='prodotto_cariche |q1 * q2|')
plt.title("Distribuizione della forza rispetto le cariche")
plt.xlabel("distanza (m^2)")
plt.ylabel("Forza (N)")
plt.grid(True)

plt.subplot(212)
plt.scatter(df_bilanciato["distanza (m)"].values, df_bilanciato["forza (N)"].values, c="mediumblue", s=40, alpha=0.6)
plt.title("Distribuizione della forza rispetto alla distanza")
plt.xlabel("distanza (m^2)")
plt.ylabel("Forza (N)")
plt.suptitle("Analisi della distribuizione della forza e della potenza delle cariche")
plt.grid(True)
plt.show()


#instanziare il modello di rete neurale, con tutti i parametri
NeuralNet=nn.Architettura(nn_layers=[2, 8, 32, 32, 8, 1], init_pesi="He", epochs=150,
                                        X_train=data_features, y_train=data_targets, learning_rate=5e-3, 
                                        ottimizzattore="Adagrad", funzione_perdita="MSE", attivazione="leaky_ReLU")
#istanziare l'oggetti per l'organizzazione e processamento dei dati
metriche=utils.Metriche(dataset=data_path, modello=NeuralNet)

#allenare e valutare il modello con la validazione incrociata
metriche.cross_validation(K_folds=5, features=data_features, targets=data_targets)
forza_elettrica=PIML.Fisica.Forza_elettrica_leggeCoulomb(q1=df["Carica_1 (C)"].values, q2=df["Carica_2 (C)"].values, dist=df["distanza (m)"].values)
Reg_coulomb=NeuralNet.migliore_modello

predizioni=Reg_coulomb.predict(inputs=metriche.x_test)
plt.figure(figsize=(10, 6))
plt.scatter(metriche.x_test[:, 1], metriche.y_test, s=25, c="mediumblue", label=f"dati dataset")
plt.scatter(metriche.x_test[:, 1], predizioni, s=25, c="orangered", label=f"predizioni modello")
plt.title("Errore Test Fold")
plt.xlabel("distanza in metri")
plt.ylabel("forza in newtons")
plt.grid(True)
plt.legend()
plt.show()


#elaborazione delle predizioni
predizioni_log=Reg_coulomb.predict(inputs=data_features)
#predizioni=10 ** predizioni_log
#Forza=df["forza (N)"].values / np.absolute(prodotto_cariche)


plt.figure(figsize=(12, 8))
plt.scatter(df_bilanciato["distanza (m)"].values, np.absolute(df_bilanciato["forza (N)"].values), c="mediumblue", alpha=0.6, label="dati puri bilanciati")
plt.scatter(df_bilanciato["distanza (m)"].values, predizioni_log, c="limegreen", label="predizioni del modello")
plt.title("Analise della Prestazione del modello")
plt.xlabel("Distanza (m^2)")
plt.ylabel("Forza (N)")
plt.grid(True)
plt.legend()
plt.show()
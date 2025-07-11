import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.Rete_Neurale_Multistrato import NeuralNetwork
from src.Tools import functions
from Tools import utils 


data_path="Datasets/fisica_Materiali.csv"
data=pd.read_csv(data_path)
print(data.head())

data_features=data[["densita","modulo_elasticita","resistenza_trazione","conduc_termica"]].values
data_labels=data["etichetta"].values

utils=utils.Processore(dataset=data_path, modello=None)
cat_indici, OneHot_criptografato=utils.codificazione_OneHot(data_categorica=data_labels)
print(f"dato categorico criptografato: {OneHot_criptografato}")
print(f"cat indici: {cat_indici}")

data_labels=OneHot_criptografato

NeuralNet=NeuralNetwork.nn_Architettura(nn_layers=[3, 6, 6, 6, 4], init_pesi="He", learning_rate=0.0003,
                                        features=data_features, targets=data_labels, epoche=1,
                                        ottimizzattore="SGD", funzione_perdita="CCE", attivazione="Tanh")
NeuralNet.Allenare()


oneHot_decriptografato=utils.decodificazione_OneHot(OneHot=OneHot_criptografato, categorie_indici=cat_indici)
print(f"dato categorico decriptografato: {oneHot_decriptografato}")



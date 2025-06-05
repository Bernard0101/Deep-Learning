import pandas as pd

from src.Rete_Neurale_Multistrato import NeuralNetwork
from src.Tools import processore
from src.Tools import functions

file_path="Datasets/fisica_MRUA.csv"
data=pd.read_csv(file_path)

print(data.head())
features=data[["Tempo (s)","Accelerazione (m/s²)","Velocita (m/s)"]].values
targets=data["Distanza (m)"].values

NeuralNet=NeuralNetwork.nn_Architettura(nn_layers=[3, 6, 6, 6, 1], init_pesi="Xavier", attivazione="")

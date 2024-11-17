import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class DatasetDilatazioneTermica:
    def __init__(self, df):
        self.dataframe=df.values
        self.materiale=['Materiale']
        self.coeff_materiale=['Coeff Dilatazione (α)']
        self.temp_iniziale=['Temperatura Iniziale (°C)']
        self.temp_finale=['Temperatura Finale (°C)']
        self.delta_L=['ΔL (m)']


class DatasetLeggiCoulomb:
    def __init__(self, df):
        self.dataframe=pd.read_csv(df)
        self.carga1=self.dataframe['Carga 1 (Coulombs)'].values
        self.carga2=self.dataframe['Carga 2 (Coulombs)'].values
        self.distanza=self.dataframe['Distanza (m)'].values
        self.forza=self.dataframe['Forza (N)'].values

    def standartizareDatti(self):
        mean=np.mean(self.dataframe, axis=0)
        deviation=np.std(self.dataframe, axis=0)
        standardize=(self.dataframe-mean)/deviation
        standardized_df = pd.DataFrame(standardize, columns=self.dataframe.columns)
        return standardized_df

    def plotDataset(self, x, y):

        #plottare relazione tra forza e distanza
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, s=10, c='teal')

        #agginugere la scala per logarimica per rendere piu faccile da vedere
        plt.yscale('log')

        #le legend per l'asse x e y
        plt.xlabel("distanza tra le cariche")
        plt.ylabel("la forza eletrica")
        plt.title("Relazione tra distanza e la forza")
        plt.grid(True, which="both", ls="-")
        plt.show()

    def PlotModeloProgress(self, x, y):

        #definire la misura e il tipo de plot
        plt.figure(figsize=(10, 6)), 
        plt.plot(x, y, label='Loss over epochs', color='teal')

        #le legend
        plt.xlabel("epochi")
        plt.ylabel("perdita")
        plt.title("progresso del imparo del modelo")
        plt.show()

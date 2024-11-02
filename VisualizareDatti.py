import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class DatasetLeggiCoulomb:
    def __init__(self, df):
        self.dataframe=df.values
        self.carga1=df['Carga 1 (Coulombs)'].values
        self.carga2=df['Carga 2 (Coulombs)'].values
        self.distanza=df['Distanza (m)'].values
        self.forza=df['Forza (N)'].values

    def standartizareDatti(self):
        mean=np.mean(self.dataframe, axis=0)
        deviation=np.std(self.dataframe, axis=0)
        standardize=(self.dataframe-mean)/deviation
        return standardize

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



esplorare=DatasetLeggiCoulomb(df=pd.read_csv("dataset_Legge_di_coulomb/Dataset_Legge_Di_Coulomb.csv"))
dataset_standardized=esplorare.standartizareDatti()
dataset_standardized=pd.DataFrame(dataset_standardized, columns=['Carga 1 (Coulombs)', 'Carga 2 (Coulombs)', 'Distanza (m)', 'Forza (N)'])
esplorare.plotDataset(x=esplorare.distanza, y=esplorare.forza)
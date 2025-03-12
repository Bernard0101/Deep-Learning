import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class processore:
    def __init__(self, df):
        self.dataframe=pd.read_csv(df)
        
    def standartizareDatti(self):
        mean=np.mean(self.dataframe, axis=0)
        deviation=np.std(self.dataframe, axis=0)
        standardize=(self.dataframe-mean)/deviation
        standardized_df=pd.DataFrame(standardize, columns=self.dataframe.columns)
        return standardized_df



class DatasetLeggiCoulomb:
    def __init__(self, df):
        self.dataframe=pd.read_csv(df)
        self.carga1=self.dataframe['Carga 1 (Coulombs)'].values
        self.carga2=self.dataframe['Carga 2 (Coulombs)'].values
        self.distanza=self.dataframe['Distanza (m)'].values
        self.forza=self.dataframe['Forza (N)'].values


    def plottare_analisi(self, x, y):

        #plottare relazione tra forza e distanza
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, s=10, c='orange')

        #agginugere la scala per logarimica per rendere piu faccile da vedere
        plt.yscale('log')

        #le legend per l'asse x e y
        plt.xlabel("distanza tra le cariche (Coulombs)")
        plt.ylabel("la forza eletrica (Newtons)")
        plt.title("Relazione tra Distanza e la Forza Elettrica")
        plt.show()

    def PlotModeloProgress(self, epochi, errori):

        #plotare la tassa di imparo del modelo
        plt.figure(figsize=(10, 6)), 
        plt.plot(epochi, errori, label='perdita tra l`epoche', color='red')

        #le legend
        plt.xlabel("epoche")
        plt.ylabel("perdita")
        plt.title("Progresso nel imparo del modelo")
        plt.show()


    def comparezioneRisultato(self, predizioni, targets):

        #creando il totale de indices
        indices = range(1, len(predizioni) + 1)        

        #plot predizioni con i targets
        plt.figure(figsize=(12, 6))


        #Grafico per le predizioni
        plt.subplot(1, 2, 1)
        plt.scatter(x=predizioni, y=indices, s=50, c='teal', alpha=0.7)
        plt.xlim(-5, 5)
        plt.title('Predizioni')

        #Grafico per i campioni
        plt.subplot(1, 2, 2)
        plt.scatter(x=targets, y=indices, s=50, c='orange', alpha=0.7)
        plt.xlim(-5, 5)
        plt.title('Campioni')

        plt.show()
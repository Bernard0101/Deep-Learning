import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



data=pd.read_csv("Dataset_Dilatazione_Termica.csv")



# Creiamo una nuova colonna per il cambiamento di temperatura
data['Cambio Temperatura'] = data['Temperatura Finale'] - data['Temperatura Iniziale']

# Grafico della variazione di lunghezza rispetto al cambiamento di temperatura
plt.figure(figsize=(10, 6))

for materiale in data['Materiale'].unique():
    subset = data[data['Materiale'] == materiale]
    plt.scatter(
        subset['Cambio Temperatura'], 
        subset['Variazione di Lunghezza'], 
        label=materiale, 
        s=100, alpha=0.7
    )

plt.title('Variazione di Lunghezza rispetto al Cambiamento di Temperatura', fontsize=16)
plt.xlabel('Cambio di Temperatura (°C)', fontsize=14)
plt.ylabel('Variazione di Lunghezza (m)', fontsize=14)
plt.legend(title='Materiale')
plt.grid(True)
plt.show()
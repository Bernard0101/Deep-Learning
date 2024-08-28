import pandas as pd
import numpy as np

# Carica il dataset
data = pd.read_csv('salary_data_with_noise.csv')

# Separa le caratteristiche (X) e l'etichetta (y)
features = data[['Age', 'Education']]
labels = data['Salary']

# Calcola la media e la deviazione standard per ciascuna caratteristica
means = features.mean()
stds = features.std()

# Standardizza le caratteristiche
features_standardized = (features - means) / stds

# Creare un nuovo DataFrame con le caratteristiche standardizzate
features_standardized_df = pd.DataFrame(features_standardized, columns=['Age', 'Education'])

# Aggiungi l'etichetta (Salary) al DataFrame standardizzato
features_standardized_df['Salary'] = labels

# Scrivi il nuovo DataFrame in un file CSV
features_standardized_df.to_csv('dataset.csv', index=False)

print("Il file 'dataset.csv' è stato scritto con successo!")

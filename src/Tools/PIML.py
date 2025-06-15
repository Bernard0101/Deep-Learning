import numpy as np

from src.Tools import functions as nn_func
from src.Tools import processore


class Fisica:
    def __init__(self, dataset):
        self.dataset=dataset
        

    def Forza_elettrica_leggeCoulomb(features):
        k=8.987e9
        carica_1=features[:, 0]
        carica_2=features[:, 1]
        distanza=features[:, 2]
        forza_elettrica=k * (carica_1 * carica_2) / distanza ** 2
        return forza_elettrica.flatten()
    
    def MAE_leggeCoulomb(self, y_pred, features):
        forza_elettrica=self.Forza_elettrica_leggeCoulomb(features=features)
        MAE_leggeColomb=nn_func.nn_functions.Loss_MAE(y_pred=y_pred, y_label=forza_elettrica)
        return MAE_leggeColomb

    def MAE_derivata_leggeCoulomb(self, y_pred, features):
        forza_elettrica=self.Forza_elettrica_leggeCoulomb(features=features)
        MAE_derivata_leggeCoulumb=nn_func.nn_functions.Loss_MAE_derivative(y_pred=y_pred, y_label=forza_elettrica)
        return MAE_derivata_leggeCoulumb


    def MSE_leggeCoulomb(self, y_pred, features):
        forza_elettrica=self.Forza_elettrica_leggeCoulomb(features=features)
        std_forza_elettrica=processore.Processore.standardizzare_data(data=forza_elettrica)
        MSE_leggeColumb=nn_func.nn_functions.Loss_MSE(y_pred=y_pred, y_label=std_forza_elettrica)
        return MSE_leggeColumb

    def MSE_derivata_leggeCoulomb(self, features, y_pred):
        forza_elettrica=self.Forza_elettrica_leggeCoulomb(features=features)
        std_forza_elettrica=processore.Processore.standardizzare_data(data=forza_elettrica)
        MSE_derivata_leggeCoulomb=nn_func.nn_functions.Loss_MSE_derivative(y_pred=y_pred, y_label=std_forza_elettrica)
        return MSE_derivata_leggeCoulomb
       

        
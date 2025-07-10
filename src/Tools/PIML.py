import numpy as np

from src.Tools import functions as nn_func
from src.Tools import processore


class Fisica:
    def __init__(self, features, targets):
        self.features=features
        self.targets=targets
        

    def Forza_elettrica_leggeCoulomb(q1, q2, dist):
        delta=3e-9
        k=8.987e9
        forza_elettrica=k * np.dot(q1, q2) / dist ** 2 + delta
        return forza_elettrica.flatten()
    
   

        
import numpy as np
import matplotlib.pyplot as plt

from src.Perceptrone import perceptron


dataset=np.array([[0, 1, 0, 1, 0, 1, 0, 1],
                  [0, 0, 1, 1, 0, 0, 1, 1],
                  [0, 0, 0, 0, 1, 1, 1, 1], 
                  [0, 0, 0, 1, 1, 0, 1, 1]]).T


target_and=np.array([0, 0, 0, 0, 0, 0, 0, 1])
target_or=np.array([0, 1, 1, 1, 1, 1, 1, 1])
target_nand=np.array([1, 1, 1, 1, 1, 1, 1, 0])
target_nor=np.array([1, 0, 0, 0, 0, 0, 0, 0])
target_xor=np.array([0, 1, 1, 1, 1, 1, 1, 0])
target_xnor=np.array([1, 0, 0, 0, 0, 0, 0, 1])



percettrone=perceptron.Perceptron(features=dataset, targets=target_xnor, learning_rate=0.5, epoche=50)
percettrone.Allenare()

for esempio in dataset:
    pred=percettrone.predict(feature=esempio)
    print(pred)


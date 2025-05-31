import numpy as np

from src.Perceptrone import perceptron


dataset=np.array([[0, 1, 0, 1, 0, 1, 0, 1],
                  [0, 0, 1, 1, 0, 0, 1, 1],
                  [0, 0, 0, 0, 1, 1, 1, 1]]).T



target_or=np.array([0, 1, 1, 1, 1, 1, 1, 1])
target_and=np.array([0, 0, 0, 0, 0, 0, 0, 1])
target_xor=np.array([0, 1, 1, 1, 1, 1, 1, 0])



percettrone=perceptron.Perceptron(features=dataset, targets=target_and, learning_rate=0.5, epoche=25)
percettrone.Allenare()

for esempio in dataset:
    pred=percettrone.predict(feature=esempio)
    print(pred)

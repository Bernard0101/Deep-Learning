import numpy as np
from perceptron import Perceptron, TrainPerceptron
features = None

#input layer perceptrons
p=Perceptron(n_features=len(features))
node=TrainPerceptron()


class MultiLayerPerceptron(Perceptron, TrainPerceptron):
    def __init__(self, hidden_layers=0):
        pass

    def input_layer(self):
        feed1=node.training_loop(p)
        feed2=node.training_loop(p)
    
    def hidden_layer(self, feed1, feed2):
        pred1 = feed1
        pred2 = feed2
        p1=p.fit(X=feed1)
        p2=p.fit(X=feed2)
        node.training_loop(p1)
        node.training_loop(p2)
        


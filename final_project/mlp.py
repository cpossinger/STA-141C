import numpy as np
from numpy.lib import gradient
from tqdm import trange
import time 
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, log_loss
import pandas as pd


np.random.seed(42)

class MLPClassifier:
    def __init__(self):
        self.weights = []
        self.bias = []
        self.activations = []
        self.z = []
        self.error = []
        self.gradient = [] 
        self.X_shape = (0,0) 
    
    def _sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_prime(self,z):
        sigmoid_z = self._sigmoid(z)
        return sigmoid_z * (1 - sigmoid_z)

    def cross_entropy_prime(self,y_true, y_pred):
        epsilon = 1e-7  # Small value to avoid division by zero
        clipped_y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)  # Clip predicted values to avoid numerical instability
        return -(y_true / clipped_y_pred - (1 - y_true) / (1 - clipped_y_pred))


    def add_input_layer(self,num_nodes):
        self.X_nodes = num_nodes

    def add_hidden_layer(self,num_nodes):
        self.activations.append(np.empty((num_nodes,1,)))
        self.z.append(np.empty(num_nodes))
        self.bias.append(np.random.randn(num_nodes,1,) * 0.01)
        if len(self.activations) == 1:
            self.weights.append(np.random.randn(self.activations[-1].shape[0],self.X_nodes ) * 0.01)
        else:
            self.weights.append(np.random.randn(self.activations[-1].shape[0],self.activations[-2].shape[0]) * 0.01)

    def add_output_layer(self,num_nodes):
        self.activations.append(np.empty((num_nodes,1)))
        self.z.append(np.empty(num_nodes))
        self.bias.append(np.random.randn(num_nodes,1) * 0.01)
        self.weights.append(np.random.randn(self.activations[-1].shape[0],self.activations[-2].shape[0]) * 0.01)
           

    def _feed_forward(self,X):

        X = X.reshape(X.shape[0],1)

        for l in range(len(self.activations)):
            if l == 0:
                z = (self.weights[l] @ X) + self.bias[l]
            else:
                z = (self.weights[l] @ self.activations[l - 1]) + self.bias[l]

            self.z[l] = z 
            self.activations[l] = self._sigmoid(z)

        
    def _back_prop(self,y_true):

        output_error = self.cross_entropy_prime(y_true, self.activations[-1]) * self._sigmoid_prime(self.z[-1])
        self.error[-1] += output_error

        for l in range(len(self.activations) - 2, -1, -1):
            delta = (self.weights[l + 1].T @ self.error[l + 1]) * self._sigmoid_prime(self.z[l])
            self.error[l] += delta

        for l in range(len(self.weights)-1,-1,-1):
            self.gradient[l] += self.error[l] @ self.activations[l - 1].T
        
    def train(self,X,y,learning_rate,m,num_epochs):

        start_time = time.time()
        self.error = [np.zeros_like(arr) for arr in self.bias]
        self.gradient = [np.zeros_like(arr) for arr in self.weights]

        for _ in range(num_epochs):

            # Copy the array to preserve the original
            remaining_rows = X.copy()

            # Continue sampling until there are no rows left
            while remaining_rows.shape[0] > 0:
                # Randomly sample rows without replacement
                sample_size = min(m, remaining_rows.shape[0])  # Set the desired sample size (e.g., 2)
                mini_batch = np.random.choice(remaining_rows.shape[0], size=sample_size, replace=False)
                
                for row in mini_batch: 

                    #self._feed_forward(np.expand_dims(X[row],axis = 1))
                    self._feed_forward(X[row])
                    self._back_prop(y[row])
               
                # Remove the sampled rows from the remaining rows
                remaining_rows = np.delete(remaining_rows, mini_batch, axis=0)

                self.weights = [weights - ((learning_rate / len(mini_batch)) * gradient ) for weights, gradient in zip(self.weights, self.gradient)]
                self.bias = [bias - ((learning_rate / len(mini_batch)) * error)  for bias, error in zip(self.bias, self.error)]
                
        total_time = time.time() - start_time 
        print("Elapsed time:", total_time, "seconds")


    def predict(self,X_test):
        predictions = []
        for row in range(X_test.shape[0]):
            self._feed_forward(X_test[row])
            if self.activations[-1][0][0] >= 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions 

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

test = MLPClassifier()

test.add_input_layer(179)
test.add_hidden_layer(5)
test.add_hidden_layer(5)
test.add_hidden_layer(5)
test.add_output_layer(1)

test.train(X_train,y_train,0.01,1000,1)
y_pred = test.predict(X_test)

print(f1_score(y_test,y_pred))




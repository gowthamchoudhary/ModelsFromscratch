import numpy as np 
from collections import Counter




class KNN:
    def __init__(self,K=3,task="classification"):
        self.K = K
        self.task = task

        self.X_train = None
        self.y_train = None
    def fit(self,x,y):
        self.X_train =np.array(x) 
        self.y_train = np.array(y)
    def calculate_distance(self,new_set):
        return np.sqrt(np.sum((np.array(new_set)-np.array(self.X_train))**2,axis=1),)
    def calculate_neighbors(self,locs):
        neighbors = []
        for i in locs:
            neighbors.append(self.y_train[i])
        return neighbors
        
    def predict_one(self,X):
        
        distances = self.calculate_distance(X)
        k_indices = np.argsort(distances)[:self.K]
        neighbors = self.calculate_neighbors(k_indices)
        
        if self.task=="classification":
            predicted_class = Counter(neighbors).most_common(1)[0][0]
            return predicted_class
        else:
            return np.mean(neighbors)
    def predict_multiple(self,X_test):
        predictions = []
        for i in X_test:
            predictions.append(self.predict_one(i))
        return predictions


      
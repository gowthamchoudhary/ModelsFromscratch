import numpy as np 
from collections import Counter
class KNN:
    def __init__(self,K=3,task="classification",weighted=True):
        self.K = K
        self.task = task
        self.weighted = weighted
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
    def calculate_weights(self,k_distances,neighbors):
        
        weights = {}
        for lable,distance in zip(neighbors,k_distances):
            weight = 1/distance
            if lable not in weights:
                weights[lable]=0
            weights[lable]+=weight
        return weights     
        
    def predict_one(self,X):
        
        distances = self.calculate_distance(X)
        k_indices = np.argsort(distances)[:self.K]
        neighbors = self.calculate_neighbors(k_indices)
        k_distances = distances[k_indices]
        
        # if self.task=="classification":
        #     # predicted_class = Counter(neighbors).most_common(1)[0][0]
        #     predicted_class = max(weights,key=weights.get)
        #     return predicted_class
        # else:
        #     predicted_value=0
        #     for i in range(self.K):
        #         predicted_value=+((weights[i]*neighbors[i])/weights[i])
        #     return predicted_value
        if self.task == "classification":
            if self.weighted:
                class_weights = self.calculate_weights(k_distances,neighbors)
                predicted_class = max(class_weights,key=class_weights.get)
                return predicted_class
            else:
                predicted_class = Counter(neighbors).most_common(1)[0][0]
        else:
            if self.weighted:
                weights = 1/k_distances
                neighbors = np.array(neighbors,dtype=float)
                predicted_value = (
                    np.sum(weights*neighbors)/np.sum(weights)

                )
                return predicted_value
            else:
                return np.mean(neighbors)
            
    def predict_multiple(self,X_test):
        predictions = []
        for i in X_test:
            predictions.append(self.predict_one(i))
        return predictions


X_train = [
    [1, 1],
    [1, 2],
    [2, 1],
    [8, 8],
    [9, 8],
    [8, 9]
]

y_train = [
    "A",
    "A",
    "A",
    "B",
    "B",
    "B"
]

X_test = [
    [2, 2],
    [8, 8]
]

knn = KNN(K=3, task="classification")

knn.fit(X_train, y_train)

predictions = knn.predict_multiple(X_test)

print(predictions)
X_train = [
    [1],
    [2],
    [3],
    [4],
    [5]
]

y_train = [
    10,
    20,
    30,
    40,
    50
]

X_test = [
    [2.5],
    [4.5]
]

knn = KNN(K=2, task="regression")

knn.fit(X_train, y_train)

predictions = knn.predict_multiple(X_test)

print(predictions)
import numpy as np


class Node:
    def __init__(self,feature=None,threshold=None,left_node=None,right_node=None,prediction=None):
        self.feature = feature
        self.threshold = threshold
        self.left_node = left_node
        self.right_node = right_node
        self.prediction = prediction
class Decision_Tree_Classifier:
    def __init__(self,max_depth=3):
        self.classes = []
        self.max_depth = max_depth
        self.root = None
    def cal_entropy(self,class_prob:list):
        class_prob = np.asarray(class_prob)
        return np.sum([-p*np.log2(p) for p in class_prob if p>0])
    def Gini(self,class_prob):
        class_prob = np.asarray(class_prob)
        return 1-np.sum(class_prob**2)
    def weighted_gini(self,left_y,right_y):
        gini_prob_left = self.Gini(self.class_prob(left_y))
        gini_prob_right = self.Gini(self.class_prob(right_y))
        l  = len(left_y)+len(right_y)
        return ((len(left_y)/l)*gini_prob_left + (len(right_y)/l)*gini_prob_right)
    def cal_Threshold(self, x_feature):
        x_feature = np.asarray(x_feature)
        x_feature = np.sort(x_feature)

        mp = [
            (x_feature[j] + x_feature[j + 1]) / 2
            for j in range(len(x_feature) - 1)
        ]

        return mp

    def class_prob(self,y):
        y = np.array(y)
        unq = np.unique(y)
        class_probabilities = []
        for i in unq:
            class_probabilities.append((y==i).sum()/len(y))
        return class_probabilities
        
    def best_split(self,x_feature,y):
        threshold_candidates = self.cal_Threshold(x_feature)
        final_gini_scores=[]
        for i in threshold_candidates:
            left = y[x_feature<i]
            right = y[x_feature>=i]
       
            w_g = self.weighted_gini(left,right)
            final_gini_scores.append(w_g)
        best_index = np.argmin(final_gini_scores)
        best_threshold = threshold_candidates[best_index]
        best_gini = final_gini_scores[best_index]
        return best_threshold , best_gini

    def find_best_split(self,x_train,y):
        best_threshold = None
        best_feature = None
        best_gini=float("inf")

        for feature in range(x_train.shape[1]):
            x_feature = x_train[:,feature]
            threshold , gini = self.best_split(x_feature,y)
            if gini<best_gini:
                best_gini = gini
                best_threshold= threshold
                best_feature = feature
        return best_feature,best_threshold,best_gini
    def majority_class(self,y):
        classes,counts = np.unique(y,return_counts=True)
        return classes[np.argmax(counts)]
    def fit(self,x_train,y_train,):
        self.x_train  = x_train
        self.y_train = y_train
        self.classes = np.unique(y_train)
        self.root = self.Build_the_tree(x_train,y_train,0)
    
    def Build_the_tree(self,X,y,depth):
        if len(np.unique(y))==1:
            return Node(prediction=y[0])
        if self.max_depth is not None  and depth>=self.max_depth:
            return Node(prediction=self.majority_class(y))
        best_feature,best_threshold,best_gini = self.find_best_split(X,y)
        left_mask = (
            X[:, best_feature] < best_threshold
        )

        right_mask = (
            X[:, best_feature] >= best_threshold
        )
        X_left = X[left_mask]
        y_left = y[left_mask]

        X_right = X[right_mask]
        y_right = y[right_mask]
        left_node = self.Build_the_tree(
            X_left,
            y_left,
            depth + 1
        )
        right_node = self.Build_the_tree(
            X_right,
            y_right,
            depth + 1
        )
        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left_node=left_node,
            right_node=right_node
        )
    def _predict_one(self, x, node):

    
        if node.prediction is not None:

            return node.prediction

       
        if x[node.feature] < node.threshold:

            return self._predict_one(
                x,
                node.left_node
            )

       
        return self._predict_one(
            x,
            node.right_node
        )

    def predict(self, X):

        X = np.asarray(X)

        predictions = []

        for x in X:

            prediction = self._predict_one(
                x,
                self.root
            )

            predictions.append(prediction)

        return np.asarray(predictions)  

X_train = np.array([
    [18, 20000,  5, 70, 1],
    [20, 22000,  6, 72, 1],
    [21, 25000,  7, 75, 1],
    [22, 28000,  8, 78, 2],
    [24, 30000, 10, 80, 2],
    [25, 35000, 11, 82, 2],
    [26, 40000, 12, 85, 3],
    [27, 42000, 13, 87, 3],
    [28, 45000, 14, 88, 4],
    [30, 48000, 15, 90, 4],
    [32, 52000, 16, 91, 5],
    [35, 60000, 18, 94, 5],
    [19, 21000,  5, 68, 1],
    [23, 27000,  9, 79, 2],
    [29, 46000, 14, 89, 4],
    [31, 50000, 15, 90, 4],
    [34, 58000, 17, 93, 5],
    [36, 65000, 19, 95, 6],
    [40, 70000, 20, 96, 6],
    [42, 75000, 21, 97, 7]
])

y_train = np.array([
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1
])


tree = Decision_Tree_Classifier(max_depth=4)
tree.fit(X_train, y_train)

X_test = np.array([
    [19, 23000,  6, 71, 1],
    [23, 29000,  9, 79, 2],
    [26, 39000, 12, 84, 3],
    [29, 47000, 14, 89, 4],
    [33, 55000, 16, 92, 5],
    [38, 68000, 19, 95, 6],
    [21, 26000,  7, 74, 1],
    [27, 43000, 13, 86, 3],
    [35, 62000, 18, 94, 5],
    [41, 72000, 21, 97, 7]
])


predictions = tree.predict(X_test)

print("Predictions:")
print(predictions)
y_test = np.array([
    0,
    0,
    1,
    1,
    1,
    1,
    0,
    1,
    1,
    1
])

print("Predicted:", predictions)
print("Actual:   ", y_test)

accuracy = np.mean(predictions == y_test)

print("Accuracy:", accuracy)
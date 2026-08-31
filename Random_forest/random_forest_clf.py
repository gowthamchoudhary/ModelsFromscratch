import numpy as np



class Node:
    def __init__(self,feature=None,threshold=None,left_node=None,right_node=None,prediction=None):
        self.feature = feature
        self.threshold = threshold
        self.left_node  = left_node
        self.right_node = right_node
        self.prediction = None
    


class DecisionTree:
    def __call__(self, max_depth=3,max_features=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.root = None
    def gini_impurity(self,y):
        classes , count = np.unique(y,return_counts=True)
        prob = count/len(y)
        return 1-np.sum(prob**2)
    def weighted_gini(self,left_y,right_y):
        total = len(left_y)+len(right_y)
        return ((len(left_y)/total)*self.gini_impurity(left_y)+(len(right_y)/total)*self.gini_impurity(right_y))
    def get_threshold(self,x_feature):
        x_arranged = np.sort(np.unique(x_feature))
        return (x_arranged[:-1]+x_arranged[:-1])/2 
    def best_split_for_feature(
            self,
            X,
            y,
            feature
    ):
        values = X[:,feature]
        thresholds = self.get_threshold(values)
        best_threshold = None
        best_gini = float("inf")
        for threshold in thresholds:
            left_mask = values<threshold
            right_mask = values>=threshold
            left_y = y[left_mask]
            right_y = y[right_mask]
            if len(left_y)==0 or len(right_y)==0:
                continue
            score = self.weighted_gini(
                left_y,
                right_y
            )
            if score>best_gini:
                best_gini = score
                best_threshold = threshold
        return best_threshold,best_gini
    def find_best_split(self, X, y):

        best_feature = None
        best_threshold = None
        best_gini = float("inf")

        n_features = X.shape[1]

        if self.max_features is None:

            feature_indices = np.arange(n_features)

        else:

            feature_indices = np.random.choice(
                n_features,
                size=self.max_features,
                replace=False
            )

        for feature in feature_indices:

            threshold, gini = self.best_split_for_feature(
                X,
                y,
                feature
            )

            if gini < best_gini:

                best_gini = gini
                best_threshold = threshold
                best_feature = feature

        return best_feature, best_threshold, best_gini
    def majority_class(self,y):
        classes,count = np.unique(y,return_counts=True)
        return classes[np.argmax(count)]
    def build_tree(self,X,y,depth):
        if len(np.unique(y))==1:
            return Node(prediction=y[0])
        if self.max_depth is not None and depth>=self.max_depth:
            return Node(prediction=self.majority_class(y)) 
        feature,threshold,_ = self.find_best_split(X,y)
        values = X[:,feature]
        left_mask = values<threshold
        right_mask = values>=threshold
        X_left = X[left_mask]
        y_left = y[left_mask]
        X_right = X[right_mask]
        y_right = y[right_mask]
        left_node = Node(
            X_left,y_left,depth+1
        )
        right_node = Node(
            X_right,y_right,depth+1
        )
        return Node(
    feature=feature,
    threshold=threshold,
    left=left_node,
    right=right_node
)
    def fit(self,X,y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.root = self.build_tree(
        X,
        y,
        depth=0
    )
    def _predict_one(
    self,
    x,
    node
):
        if node.prediction is not None:
           return node.prediction
        if x[node.feature] < node.threshold:
            return self._predict_one(
    x,
    node.left
)       
    def predict(
        self,
        X
    ):

        X = np.asarray(X)

        return np.array([
            self._predict_one(
                x,
                self.root
            )
            for x in X
        ])

        
class RandomForest:
    def __init__(self,n_trees=0,max_depth=0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
    def Bootstrap_sample(self,X,y):
        n_samples = X.shape[0]
        indices = np.random.choice(
            n_samples,
            size=n_samples
            ,
            replace=True
        )
        return X[indices],y[indices]
    def fit(self,X,y):
        trees = []
        for _ in range(self.n_trees):

            X_sample, y_sample = self.bootstrap_sample(X, y)

            tree = DecisionTree(
                max_depth=self.max_depth,
                max_features=self.max_features
            )

            tree.fit(X_sample, y_sample)

            self.trees.append(tree)
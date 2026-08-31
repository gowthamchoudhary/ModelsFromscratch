import numpy as np


class Node:

    def __init__(
        self,
        feature=None,
        threshold=None,
        left_node=None,
        right_node=None,
        prediction=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left_node = left_node
        self.right_node = right_node
        self.prediction = prediction


class DecisionTree:

    def __init__(self, max_depth=3, max_features=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.root = None

    def gini_impurity(self, y):

        classes, count = np.unique(
            y,
            return_counts=True
        )

        prob = count / len(y)

        return 1 - np.sum(prob ** 2)

    def weighted_gini(self, left_y, right_y):

        total = len(left_y) + len(right_y)

        return (
            (len(left_y) / total) *
            self.gini_impurity(left_y)
            +
            (len(right_y) / total) *
            self.gini_impurity(right_y)
        )

    def get_threshold(self, x_feature):

        x_sorted = np.sort(
            np.unique(x_feature)
        )

        return (
            x_sorted[:-1] +
            x_sorted[1:]
        ) / 2

    def best_split_for_feature(
        self,
        X,
        y,
        feature
    ):

        values = X[:, feature]

        thresholds = self.get_threshold(values)

        best_threshold = None
        best_gini = float("inf")

        for threshold in thresholds:

            left_mask = values < threshold
            right_mask = values >= threshold

            left_y = y[left_mask]
            right_y = y[right_mask]

            if (
                len(left_y) == 0
                or len(right_y) == 0
            ):
                continue

            score = self.weighted_gini(
                left_y,
                right_y
            )

            if score < best_gini:

                best_gini = score
                best_threshold = threshold

        return best_threshold, best_gini

    def find_best_split(self, X, y):

        best_feature = None
        best_threshold = None
        best_gini = float("inf")

        n_features = X.shape[1]

        if self.max_features is None:

            feature_indices = np.arange(
                n_features
            )

        else:

            feature_indices = np.random.choice(
                n_features,
                size=min(
                    self.max_features,
                    n_features
                ),
                replace=False
            )

        for feature in feature_indices:

            threshold, gini = (
                self.best_split_for_feature(
                    X,
                    y,
                    feature
                )
            )

            if (
                threshold is not None
                and gini < best_gini
            ):

                best_feature = feature
                best_threshold = threshold
                best_gini = gini

        return (
            best_feature,
            best_threshold,
            best_gini
        )

    def majority_class(self, y):

        classes, count = np.unique(
            y,
            return_counts=True
        )

        return classes[
            np.argmax(count)
        ]

    def build_tree(self, X, y, depth):

        if len(np.unique(y)) == 1:

            return Node(
                prediction=y[0]
            )

        if (
            self.max_depth is not None
            and depth >= self.max_depth
        ):

            return Node(
                prediction=self.majority_class(y)
            )

        feature, threshold, _ = (
            self.find_best_split(X, y)
        )

        if feature is None:

            return Node(
                prediction=self.majority_class(y)
            )

        values = X[:, feature]

        left_mask = values < threshold
        right_mask = values >= threshold

        X_left = X[left_mask]
        y_left = y[left_mask]

        X_right = X[right_mask]
        y_right = y[right_mask]

        left_node = self.build_tree(
            X_left,
            y_left,
            depth + 1
        )

        right_node = self.build_tree(
            X_right,
            y_right,
            depth + 1
        )

        return Node(
            feature=feature,
            threshold=threshold,
            left_node=left_node,
            right_node=right_node
        )

    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)

        self.root = self.build_tree(
            X,
            y,
            depth=0
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

        return np.array([
            self._predict_one(
                x,
                self.root
            )
            for x in X
        ])


class RandomForest:

    def __init__(
        self,
        n_trees=10,
        max_depth=3,
        max_features=None
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def bootstrap_sample(self, X, y):

        n_samples = X.shape[0]

        indices = np.random.choice(
            n_samples,
            size=n_samples,
            replace=True
        )

        return (
            X[indices],
            y[indices]
        )

    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)

        self.trees = []

        for _ in range(self.n_trees):

            X_sample, y_sample = (
                self.bootstrap_sample(X, y)
            )

            tree = DecisionTree(
                max_depth=self.max_depth,
                max_features=self.max_features
            )

            tree.fit(
                X_sample,
                y_sample
            )

            self.trees.append(tree)

    def predict(self, X):

        X = np.asarray(X)

        all_predictions = []

        for tree in self.trees:

            predictions = tree.predict(X)

            all_predictions.append(
                predictions
            )

        all_predictions = np.array(
            all_predictions
        )

        final_predictions = []

        for i in range(X.shape[0]):

            tree_predictions = (
                all_predictions[:, i]
            )

            classes, counts = np.unique(
                tree_predictions,
                return_counts=True
            )

            prediction = classes[
                np.argmax(counts)
            ]

            final_predictions.append(
                prediction
            )

        return np.array(
            final_predictions
        )



X_train = np.array([
    [10, 20, 30, 40],
    [12, 22, 32, 42],
    [15, 25, 35, 45],
    [18, 28, 38, 48],
    [20, 30, 40, 50],

    [50, 60, 70, 80],
    [52, 62, 72, 82],
    [55, 65, 75, 85],
    [58, 68, 78, 88],
    [60, 70, 80, 90],

    [90, 80, 70, 60],
    [88, 78, 68, 58],
    [85, 75, 65, 55],
    [82, 72, 62, 52],
    [80, 70, 60, 50]
])

y_train = np.array([
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2
])


X_test = np.array([
    [11, 21, 31, 41],
    [17, 27, 37, 47],
    [54, 64, 74, 84],
    [59, 69, 79, 89],
    [89, 79, 69, 59],
    [83, 73, 63, 53],
    [30, 40, 50, 60],
    [75, 65, 55, 45]
])

y_test = np.array([
    0,
    0,
    1,
    1,
    2,
    2,
    1,
    2
])



model = RandomForest(
    n_trees=20,
    max_depth=4,
    max_features=2
)

model.fit(
    X_train,
    y_train
)

predictions = model.predict(
    X_test
)

print("Predictions:")
print(predictions)

print("\nActual:")
print(y_test)

accuracy = np.mean(
    predictions == y_test
)

print("\nAccuracy:", accuracy)
# Decision Tree Classifier From Scratch

[decision_tree.py](decision_tree.py) implements a binary-split classification tree with NumPy. It recursively selects the feature and threshold that produce the lowest weighted Gini impurity.

## How It Works

For each feature, the classifier evaluates midpoints between sorted feature values. A split sends values below the threshold left and all other values right. Tree growth stops when a node contains one class or reaches `max_depth`; the node then predicts its majority class.

```python
from decision_tree import Decision_Tree_Classifier

tree = Decision_Tree_Classifier(max_depth=4)
tree.fit(X_train, y_train)
predictions = tree.predict(X_test)
```

`X_train` and `X_test` should be numeric arrays shaped `(n_samples, n_features)`. `y_train` contains one class label per sample. The script includes a small numeric classification example and prints its predictions and accuracy.

## Run

```bash
pip install numpy
python decision_tree.py
```

## Key Concepts

- Gini impurity measures how mixed a node's labels are.
- Weighted Gini impurity compares the quality of candidate splits.
- `max_depth` limits tree growth and helps control overfitting.
- Leaf nodes return a class label during prediction.

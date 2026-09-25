# Random Forest Classifier From Scratch

[random_forest_clf.py](random_forest_clf.py) builds a classification random forest from scratch with NumPy. Each tree is trained on a bootstrap sample and can consider a random subset of features at each split. The forest returns the majority vote from all trees.

## Usage

```python
from random_forest_clf import RandomForest

model = RandomForest(n_trees=20, max_depth=4, max_features=2)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

`X_train` and `X_test` should be numeric arrays shaped `(n_samples, n_features)`, and `y_train` should have one class label per sample.

## Parameters

| Parameter | Meaning |
| --- | --- |
| `n_trees` | Number of decision trees to train. |
| `max_depth` | Maximum depth of each tree. |
| `max_features` | Number of randomly chosen features evaluated at each split; use `None` for all features. |

## Implementation Details

- `bootstrap_sample` samples training rows with replacement for every tree.
- `DecisionTree` searches threshold midpoints and minimizes weighted Gini impurity.
- `predict` gathers the prediction from every tree and selects the most frequent class.

## Run

```bash
pip install numpy
python random_forest_clf.py
```

The script contains a three-class numeric example and prints predictions, labels, and accuracy.

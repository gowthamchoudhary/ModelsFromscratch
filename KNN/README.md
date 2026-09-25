# K-Nearest Neighbors From Scratch

This project is a from-scratch implementation of K-Nearest Neighbors using Python and NumPy. KNN is a distance-based supervised learning algorithm that predicts the class of a new point by looking at the closest labeled examples.

The folder contains two implementations:

- [KNN_from_scratch.ipynb](KNN_from_scratch.ipynb) introduces KNN classification with a visual example.
- [knn_classification_and_regression.py](knn_classification_and_regression.py) provides a reusable `KNN` class for classification and regression, with optional inverse-distance weighting.

## Project Overview

The notebook uses a small manually created 2D dataset with two classes:

- `Blue`
- `Red`

Each point is represented as a list of numeric coordinates. A new point is compared with all existing points, and the model predicts the class that appears most often among the nearest neighbors.

## What the Notebook Covers

- Importing NumPy and Matplotlib
- Creating labeled sample points manually
- Implementing Euclidean distance from scratch
- Building a `KNearestNeighbors` class
- Storing training points with a `fit` method
- Calculating distances from a new point to all known points
- Sorting points by distance
- Selecting the nearest `k` neighbors
- Using majority voting with `Counter`
- Predicting the class label for a new point

## Maths Behind the Model

KNN compares points using a distance formula. The notebook uses Euclidean distance:

```text
distance = sqrt(sum((point1 - point2)^2))
```

For a new point, the algorithm:

1. Calculates the distance to every labeled point
2. Sorts all distances from smallest to largest
3. Takes the nearest `k` points
4. Counts the class labels of those neighbors
5. Returns the most common class

The default value used in the class is:

```text
k = 3
```

## How to Run

1. Open `KNN_from_scratch.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab and run its cells from top to bottom, or run the Python implementation:

```bash
python knn_classification_and_regression.py
```

2. The script demonstrates both classification and regression.

To use the script's model in your own code:

```python
from knn_classification_and_regression import KNN

model = KNN(K=3, task="classification", weighted=True)
model.fit(X_train, y_train)
predictions = model.predict_multiple(X_test)
```

Use `task="regression"` to return a numeric average of the neighbors instead of a class vote.

## Requirements

Install the required libraries with:

```bash
pip install numpy matplotlib
```

## What I Learned

This project helps explain the basics of KNN:

- How distance-based classification works
- Why Euclidean distance is useful for comparing numeric points
- How `k` controls the number of neighbors used for prediction
- How majority voting turns neighbor labels into a final class
- Why KNN does not train parameters like gradient-based models

# K-Nearest Neighbors From Scratch

This project is a from-scratch implementation of K-Nearest Neighbors using Python and NumPy. KNN is a distance-based supervised learning algorithm that predicts the class of a new point by looking at the closest labeled examples.

The implementation is in [KNN_from_scratch.ipynb](KNN_from_scratch.ipynb).

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

1. Open `KNN_from_scratch.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. Run the cells from top to bottom.
3. Check the printed prediction for the new point.

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

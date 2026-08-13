# Models From Scratch

This repository contains machine learning models implemented from scratch using Python and NumPy. Each model has its own folder with the notebook, supporting files, and a dedicated README explaining the idea, maths, and how to run it.

## Projects

| Model | Folder | Description |
| --- | --- | --- |
| Linear Regression From Scratch | [Linear_Regression_From_Scratch](Linear_Regression_From_Scratch) | Predicts exam scores from study hours using gradient descent and the closed-form normal equation. |
| Logistic Regression From Scratch | [Logistic_Regression](Logistic_Regression) | Classifies breast cancer samples using sigmoid activation and gradient descent. |
| Ridge Regression From Scratch | [RidgeRegression_From_Scratch](RidgeRegression_From_Scratch) | Implements ridge regression with L2 regularization using the closed-form solution and compares it with scikit-learn. |
| Neural Network From Scratch | [Neural_Network_From_Scratch](Neural_Network_From_Scratch) | Classifies handwritten digits using a simple feedforward neural network built with NumPy. |
| K-Nearest Neighbors From Scratch | [KNN](KNN) | Classifies a new point by calculating Euclidean distances and using majority voting among nearest neighbors. |
| Naive Bayes From Scratch | [Naive_Bayes](Naive_Bayes) | Classifies text sentiment using class priors, word likelihoods, and Laplace smoothing. |
| Gaussian Naive Bayes From Scratch | [Gaussian_Naive_Bayes](Gaussian_Naive_Bayes) | Classifies breast cancer samples using class-wise Gaussian likelihoods, priors, means, and variances. |

## Repository Structure

```text
ModelsFromscratch/
+-- Linear_Regression_From_Scratch/
|   +-- linearRegressionFromScratch.ipynb
|   +-- closed_form_linearRegression.ipynb
|   +-- study_scores_noisy_100.csv
|   +-- README.md
+-- Logistic_Regression/
|   +-- logistic_from_scratch.ipynb
|   +-- README.md
+-- RidgeRegression_From_Scratch/
|   +-- RidgeRegressionFromScratch.ipynb
|   +-- README.md
+-- Neural_Network_From_Scratch/
|   +-- NeuralNetworkFromScracth.ipynb
|   +-- README.md
+-- KNN/
|   +-- KNN_from_scratch.ipynb
|   +-- knn_classification_and_regression.py
|   +-- README.md
+-- Naive_Bayes/
|   +-- naive_bayes_classification.py
|   +-- README.md
+-- Gaussian_Naive_Bayes/
|   +-- Gaussian_Naive_Bayes.ipynb
|   +-- README.md
+-- README.md
```

## Requirements

The notebooks and scripts use common Python data science libraries:

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn

Install them with:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## How to Use

1. Open the folder for the model you want to study.
2. Read that folder's `README.md` for the model explanation.
3. Open the notebook in Jupyter Notebook, JupyterLab, or Google Colab, or run the Python script from that folder.
4. Run the notebook cells from top to bottom or execute the script with Python.

## Goal

The goal of this repository is to understand how machine learning models work internally by building the core training steps manually instead of depending on high-level machine learning frameworks.

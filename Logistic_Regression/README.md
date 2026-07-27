# Logistic Regression From Scratch

This project is a from-scratch implementation of logistic regression using Python and NumPy. Logistic regression is used for binary classification, where the model predicts the probability that an input belongs to one of two classes.

The implementation is in [logistic_from_scratch.ipynb](logistic_from_scratch.ipynb).

## Project Overview

The notebook trains a logistic regression model on the Breast Cancer Wisconsin dataset from scikit-learn. The model learns from medical measurement features and predicts whether each sample belongs to one of two target classes.

Unlike linear regression, logistic regression does not predict a continuous number directly. It calculates a linear score and passes it through the sigmoid function to convert the result into a probability:

```text
probability = sigmoid(Xw + b)
```

Where:

- `X` is the feature matrix
- `w` contains the learned feature weights
- `b` is the intercept or bias term
- `sigmoid` converts the score into a value between `0` and `1`

## What the Notebook Covers

- Importing NumPy
- Implementing the sigmoid function manually
- Computing logistic regression gradients with NumPy
- Adding a bias/intercept column manually
- Updating model parameters with gradient descent
- Loading the Breast Cancer Wisconsin dataset from scikit-learn
- Splitting the data into training and testing sets
- Scaling features with `StandardScaler`
- Predicting probabilities and class labels
- Evaluating model performance with accuracy score

## Maths Behind the Model

The model first computes a linear score:

```text
z = Xtheta
```

The sigmoid function converts this score into a probability:

```text
sigmoid(z) = 1 / (1 + exp(-z))
```

The prediction rule uses a threshold:

```text
prediction = 1 if probability >= 0.5
prediction = 0 if probability < 0.5
```

Gradient descent updates the parameters step by step:

```text
theta = theta - alpha * gradient
```

The gradient used in the notebook is:

```text
gradient = (1 / m) * X^T * (sigmoid(Xtheta) - y)
```

Where:

- `theta` contains the intercept and feature weights
- `alpha` is the learning rate
- `m` is the number of training examples
- `X^T` means the transpose of `X`
- `y` contains the true labels

## How to Run

1. Open `logistic_from_scratch.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. Install the required libraries.
3. Run the notebook cells from top to bottom.
4. Check the printed train and test accuracy values.

## Requirements

Install the required libraries with:

```bash
pip install numpy scikit-learn
```

## What I Learned

This project helps explain the basics of classification:

- How logistic regression turns a linear model into a classifier
- Why the sigmoid function is useful for probabilities
- How a probability threshold becomes a class prediction
- Why feature scaling helps gradient descent
- How gradients update model parameters
- How to evaluate classification accuracy on train and test data

# Gaussian Naive Bayes From Scratch

This project is a from-scratch implementation of Gaussian Naive Bayes using Python, NumPy, Pandas, and scikit-learn for the dataset and evaluation. Gaussian Naive Bayes is a probabilistic classification algorithm that works well with continuous numeric features by assuming each feature follows a Gaussian distribution inside each class.

The implementation is in [Gaussian_Naive_Bayes.ipynb](Gaussian_Naive_Bayes.ipynb).

## Project Overview

The notebook uses the breast cancer dataset from scikit-learn. Each sample contains numeric measurements from a breast mass, and the model predicts one of two target classes:

- `0`
- `1`

The model calculates the mean, variance, and prior probability for each class, then uses Gaussian probability density values to score new samples.

## What the Notebook Covers

- Importing NumPy, Pandas, and scikit-learn utilities
- Loading the breast cancer classification dataset
- Inspecting the dataset with Pandas
- Splitting data into training and testing sets
- Building a `GaussianNaiveBayes` class
- Calculating class-wise means for every feature
- Calculating class-wise variances for every feature
- Calculating class prior probabilities
- Computing Gaussian log probabilities
- Predicting labels with the highest posterior score
- Measuring accuracy with `accuracy_score`

## Maths Behind the Model

Gaussian Naive Bayes uses Bayes' theorem:

```text
P(class | x) = P(class) * P(x | class) / P(x)
```

For classification, the denominator is the same for every class, so the model compares:

```text
score = P(class) * P(feature1 | class) * P(feature2 | class) * ... * P(featureN | class)
```

For continuous features, each feature likelihood is calculated with the Gaussian probability density function:

```text
P(x | class) = 1 / sqrt(2 * pi * variance) * exp(-((x - mean)^2) / (2 * variance))
```

The notebook uses log probabilities to avoid very small probability values:

```text
log_score = log(P(class)) + sum(log(P(feature | class)))
```

## How to Run

1. Open `Gaussian_Naive_Bayes.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. Run the cells from top to bottom.
3. Check the printed accuracy score for the test set.

## Requirements

Install the required libraries with:

```bash
pip install numpy pandas scikit-learn
```

## What I Learned

This project helps explain the basics of Gaussian Naive Bayes:

- How numeric features can be modeled with class-wise Gaussian distributions
- How priors represent the frequency of each class in the training data
- How means and variances summarize each feature for each class
- Why log probabilities are useful for numerical stability
- How a probabilistic classifier can make predictions without gradient descent

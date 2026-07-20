# Ridge Regression From Scratch

This project is a from-scratch implementation of ridge regression using Python and NumPy. Ridge regression is an extension of linear regression that adds L2 regularization to reduce overfitting and make the model more stable when features are correlated.

The implementation is in [RidgeRegressionFromScratch.ipynb](RidgeRegressionFromScratch.ipynb).

## Project Overview

The notebook trains a ridge regression model on the California Housing dataset from scikit-learn. It builds the model manually with NumPy, then compares the result with scikit-learn's `Ridge` implementation.

The model learns a linear relationship between multiple housing features and the target house value:

```text
y = Xw + b
```

Where:

- `X` is the feature matrix
- `y` is the target value
- `w` contains the learned coefficients
- `b` is the intercept

## What the Notebook Covers

- Loading the California Housing dataset
- Creating a custom `RidgeRegression` class
- Adding a bias/intercept column manually
- Applying L2 regularization to the model coefficients
- Excluding the intercept from the regularization penalty
- Solving for the parameters with the closed-form ridge equation
- Making predictions with the learned coefficients
- Evaluating predictions with `r2_score`
- Comparing the from-scratch model with scikit-learn's `Ridge`

## Maths Behind the Model

Linear regression finds parameters using:

```text
theta = (X^T X)^-1 X^T y
```

Ridge regression adds a regularization term:

```text
theta = (X^T X + alpha * I)^-1 X^T y
```

Where:

- `theta` contains the intercept and coefficients
- `X` is the feature matrix with a bias column
- `y` is the target vector
- `alpha` controls the strength of regularization
- `I` is the identity matrix
- `X^T` means the transpose of `X`

In the notebook, the intercept is not regularized:

```text
I[0, 0] = 0
```

This keeps the regularization focused on the feature coefficients.

## How to Run

1. Open `RidgeRegressionFromScratch.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. Install the required libraries.
3. Run the notebook cells from top to bottom.
4. Compare the from-scratch model's R2 score with scikit-learn's Ridge model.

## Requirements

Install the required libraries with:

```bash
pip install numpy scikit-learn
```

## What I Learned

This project helps explain regularized regression:

- How ridge regression builds on linear regression
- Why L2 regularization helps control large coefficients
- How the closed-form equation changes when regularization is added
- Why the intercept is usually excluded from regularization
- How to compare a manual NumPy implementation with scikit-learn

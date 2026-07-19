# Linear Regression From Scratch

This project is a from-scratch implementation of simple linear regression using Python, NumPy, Pandas, and Matplotlib. It shows two ways to fit a regression line that predicts exam scores from study hours: gradient descent and the closed-form normal equation.

The gradient descent implementation is in [linearRegressionFromScratch.ipynb](linearRegressionFromScratch.ipynb), the closed-form solution is in [closed_form_linearRegression.ipynb](closed_form_linearRegression.ipynb), and the dataset is [study_scores_noisy_100.csv](study_scores_noisy_100.csv).

## Project Overview

The dataset contains two columns:

```text
Hours_Studied
Exam_Score
```

The model learns a straight-line relationship between study time and exam score:

```text
y = mX + b
```

Where:

- `X` is the number of hours studied
- `y` is the predicted exam score
- `m` is the slope
- `b` is the intercept

## What the Notebook Covers

### Gradient Descent Notebook

- Loading the dataset with Pandas
- Creating a mean squared error loss function
- Implementing gradient descent manually
- Updating the slope and intercept over multiple epochs
- Plotting the dataset and learned regression line

### Closed-Form Solution Notebook

- Building the feature matrix for linear regression
- Adding the bias/intercept column manually
- Solving for the best parameters directly with the normal equation
- Comparing the fitted line with the dataset visually

## Maths Behind the Model

For each data point, the model predicts:

```text
prediction = m * x + b
```

The error is measured using squared error:

```text
error = (actual - prediction) ^ 2
```

Gradient descent updates `m` and `b` step by step:

```text
m = m - learning_rate * m_gradient
b = b - learning_rate * b_gradient
```

In the notebook, the model uses:

```text
learning_rate = 0.0001
epochs = 1000
```

The closed-form solution calculates the best parameters directly:

```text
theta = (X^T X)^-1 X^T y
```

Where:

- `X` is the feature matrix with a bias column
- `y` is the target vector
- `theta` contains the intercept and slope
- `X^T` means the transpose of `X`

## How to Run

1. Open `linearRegressionFromScratch.ipynb` for gradient descent, or `closed_form_linearRegression.ipynb` for the direct closed-form solution.
2. Make sure `study_scores_noisy_100.csv` is available in the same folder, or update the CSV path in the notebook.
3. Run the notebook cells from top to bottom.
4. View the final slope, intercept, scatter plot, and regression line.

## Requirements

Install the required libraries with:

```bash
pip install numpy pandas matplotlib
```

## What I Learned

This project helps explain the basics of regression:

- How a line can model a relationship between two variables
- How loss measures prediction error
- How gradients point toward better parameter values
- How gradient descent improves the model over time
- How the closed-form solution finds the best-fit line directly
- How to visualize the fitted line against real data

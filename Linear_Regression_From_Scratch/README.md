# Linear Regression From Scratch

This project is a from-scratch implementation of simple linear regression using Python, NumPy, Pandas, and Matplotlib. The notebook trains a line to predict exam scores from study hours using gradient descent.

The implementation is in [linearRegressionFromScratch.ipynb](linearRegressionFromScratch.ipynb), and the dataset is [study_scores_noisy_100.csv](study_scores_noisy_100.csv).

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

- Loading the dataset with Pandas
- Creating a mean squared error loss function
- Implementing gradient descent manually
- Updating the slope and intercept over multiple epochs
- Plotting the dataset and learned regression line

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

## How to Run

1. Open `linearRegressionFromScratch.ipynb`.
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
- How to visualize the fitted line against real data

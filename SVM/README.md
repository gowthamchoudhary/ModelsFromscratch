# Support Vector Machines From Scratch

This folder contains three Support Vector Machine (SVM) implementations. All of them expect a numeric feature matrix `X` with shape `(n_samples, n_features)` and labels encoded as `-1` and `1`.

## Implementations

| File | Approach | Dependencies |
| --- | --- | --- |
| [svm_from_scratch_soft_margin_gd.py](svm_from_scratch_soft_margin_gd.py) | Soft-margin SVM trained with batch gradient descent on a hinge-loss objective | NumPy |
| [svm_from_scratch_poly_kernel.py](svm_from_scratch_poly_kernel.py) | Polynomial-kernel SVM using a kernelized formulation to model nonlinear decision boundaries | NumPy |
| [svm_from_scratch_hardmargin_no_gradient.py](svm_from_scratch_hardmargin_no_gradient.py) | Hard-margin SVM formulated as a constrained optimization problem | NumPy, SciPy |

## Soft-Margin SVM

`SVM_SOFT` learns a weight vector `w` and bias `b`. Samples that violate the margin contribute to the gradient, while `C` controls the penalty for those violations.

```python
from svm_from_scratch_soft_margin_gd import SVM_SOFT

model = SVM_SOFT(learning_rate=0.001, C=1, epochs=5000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

The decision rule is `sign(X @ w + b)`, returned as `-1` or `1`.

## Polynomial-Kernel SVM

The polynomial-kernel version extends the basic SVM idea by replacing the linear dot product with a polynomial kernel, which allows the model to capture nonlinear relationships in the data.

```python
from svm_from_scratch_poly_kernel import SVM_poly_kernel

model = SVM_poly_kernel(degree=3, C=1.0, learning_rate=0.01, epochs=2000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Hard-Margin SVM

`SVM_hard_margin` minimizes `0.5 * ||w||^2` subject to every training point satisfying `y_i * (w @ x_i + b) >= 1`. It uses SciPy's SLSQP optimizer rather than gradient descent.

```python
from svm_from_scratch_hardmargin_no_gradient import SVM_hard_margin

model = SVM_hard_margin().fit(X_train, y_train)
predictions = model.predict(X_test)
```

The current constraint expression in the script needs `y * scores` in place of `y(scores)` before this example can run. The soft-margin script includes a complete runnable demonstration.

## Run

```bash
pip install numpy scipy
python svm_from_scratch_soft_margin_gd.py
```

The soft-margin script prints the learned parameters, predictions, labels, and training accuracy.

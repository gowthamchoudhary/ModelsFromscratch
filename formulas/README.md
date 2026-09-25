# Euclidean Distance

[euclidean_distance.py](euclidean_distance.py) provides a NumPy implementation of Euclidean distance between two equal-length numeric points.

```python
from euclidean_distance import euclidean_distance

distance = euclidean_distance([4, 5, 6, 7], [6, 0, 2, 4])
```

The calculation is:

```text
sqrt(sum((point1 - point2)^2))
```

This measure is used by distance-based models such as K-Nearest Neighbors. The script also compares its result with SciPy's `scipy.spatial.distance.euclidean`.

## Run

```bash
pip install numpy scipy
python euclidean_distance.py
```

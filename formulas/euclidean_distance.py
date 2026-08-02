import numpy as np
from scipy.spatial.distance import euclidean


def euclidean_distance(point1,point2):
    return np.sqrt(np.sum((np.array(point1)-np.array(point2))**2))
point_a = (4,5,6,7)
point_b = (6,0,2,4)
print(euclidean_distance(point_a,point_b))
print(euclidean(point_a,point_b))

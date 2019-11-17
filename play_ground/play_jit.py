import time

import numpy as np
from numba import vectorize, float64, int64, guvectorize, f8

"""
A = [ 1, 2, 3 ]
    [ 4, 5, 6 ]
    [ 7, 8, 9 ]

B = [ 9, 8, 7 ]
    [ 6, 5, 4 ]
    [ 3, 2, 1 ]
    
C = [10, 11, 12]
    [13, 14, 15]
    [16, 17, 18]
"""

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
C = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])
I = np.array([1, 1, 1])


@vectorize([float64(int64, int64),
            int64(int64, int64)])
def np_array_dot(m1, m2):
    return m1 * m2


@vectorize([float64(int64, int64)])
def np_array_divide(m1, m2):
    return m1 / m2


@guvectorize([f8[:, :], f8[:, :], f8[:, :], f8[:, :]], "(n, n), (n, n) -> (n, n), (n, n)")
def np_dot_divide(m1, m2):
    return m1 * m2, m1 / m2


if __name__ == '__main__':
    start = time.time()
    D, E = np_dot_divide(A, B)
    end = time.time()
    print(f'np_array_dot 1: {end - start}')

    start = time.time()
    D1, E1 = np_dot_divide(B, C)
    end = time.time()
    print(f'np_array_dot 2: {end - start}')

    start1 = time.time()
    D2, E2 = A * B, A / B
    end1 = time.time()
    print(f'np * : {end1 - start1}')
    print(f'{D == D1}')




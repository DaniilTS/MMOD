import numpy as np
from collections import Counter


def random():
    return np.random.rand()


def build_dsv(P, A, B):
    q = np.sum(P, axis=1)  # 0-сумма по столбцам, 1- по строкам

    n, m = P.shape
    
    l = []
    for i in range(1, n + 1):
        l.append(sum(q[0:i]))

    k = np.searchsorted(l, random())
    x1 = A[k]

    rs = np.zeros(m)
    for i in range(1, m + 1):
        rs[i - 1] = rs[i - 2] + P[k, i - 1]

    s = np.searchsorted(rs, random() * rs[-1])
    x2 = B[s]
    return x1, x2


def build_empiric_matrix(P, A, B, amount_of_experiments):
    empiric_matrix = np.zeros(P.shape)
    discrete_values_array = Counter([build_dsv(P, A, B) for _ in range(amount_of_experiments)])
    for (x1, x2), counter in discrete_values_array.items():
        empiric_matrix[list(A).index(x1), list(B).index(x2)] = counter / amount_of_experiments

    return empiric_matrix


if __name__ == '__main__':
    P = np.array([[0.1, 0.4],
                  [0.2, 0.1],
                  [0.1, 0.1]])
    A = np.array([1, 2, 4])
    B = np.array([1, 3])

    print('Матрица вероятности P:\n', P)
    empiric_matrix = build_empiric_matrix(P, A, B, amount_of_experiments=10000)
    print('Эмпирическая матрица при кол-ве эксперементов=10000\n', empiric_matrix)


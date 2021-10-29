import numpy as np


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


if __name__ == '__main__':
    P = np.array([[0.1, 0.4],
                  [0.2, 0.1],
                  [0.1, 0.1]])
    A = np.array([1, 2, 4])
    B = np.array([1, 3])

    print('Матрица вероятности P:\n', P)

    dsv = build_dsv(P, A, B)

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import scipy.stats as stats
import math


fig, axs = plt.subplots(2)


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


def find_empiric_matrix(P, A, B, amount_of_experiments):
    empiric_matrix = np.zeros(P.shape)
    discrete_values_array = Counter([build_dsv(P, A, B) for _ in range(amount_of_experiments)])
    for (x1, x2), counter in discrete_values_array.items():
        empiric_matrix[list(A).index(x1), list(B).index(x2)] = counter / amount_of_experiments
    return empiric_matrix


def draw_histogram(A, B, empiric_matrix):
    x1_probability = np.sum(empiric_matrix, axis=1)
    x2_probability = np.sum(empiric_matrix, axis=0)
    color = (1, 0.2, 0.2, 1)
    axs[0].bar(A, x1_probability, color=color)
    axs[1].bar(B, x2_probability, color=color)
    axs[0].legend(['X1'])
    axs[1].legend(['X2'])
    plt.show()


def pearson_criterion(theoretical_matrix, empiric_matrix, n):
    chi2 = n * np.sum((empiric_matrix - theoretical_matrix) ** 2 / theoretical_matrix)
    chi2_value = stats.chi2.ppf(0.97, theoretical_matrix.size - 1)
    print('\nКритерий пирсона {}'.format('пройден успешно' if chi2 < chi2_value else 'провален'))


def build_x1_x2(P, A, B, n):
    x = [build_dsv(P, A, B) for i in range(n)]
    x1 = [x[i][0] for i in range(n)]
    x2 = [x[i][1] for i in range(n)]
    return x1, x2


def point_estimate_M(x, n):
    return math.fsum(x) / n


def point_estimate_D(x, n, M_estimate):
    return 1 / (n - 1) * math.fsum(list(map(lambda xi: (xi - M_estimate) ** 2, x)))


def intervals_for_M(n, D_estimate, M_estimate):
    arr = stats.t(n).rvs(1000000)
    probations = [0.95, 0.98, 0.99]
    deltas = [stats.mstats.mquantiles(arr, prob=probation) * math.sqrt(D_estimate / (n - 1)) for probation in probations]
    intervals = [[M_estimate - delta_n, M_estimate + delta_n] for delta_n in deltas]
    for probation, interval in zip(probations, intervals):
        print('Доверительный интервал для мат. ожидания при доверительной вероятности {}: ( {}, {} )'.format(probation, interval[0][0], interval[1][0]))
    return intervals


if __name__ == '__main__':
    P = np.array([[0.2, 0.3],
                  [0.1, 0.2],
                  [0.1, 0.1]])
    A = np.array([1, 2, 4])
    B = np.array([1, 3])
    n = 100

    print('Теоретическая матрица P:\n', P)

    empiric_matrix = find_empiric_matrix(P, A, B, n)
    print('Эмпирическая матрица при кол-ве эксперементов = {}\n'.format(n), empiric_matrix)
    # draw_histogram(A, B, empiric_matrix)
    pearson_criterion(P, empiric_matrix, n)

    x1, x2 = build_x1_x2(P, A, B, n)
    pe_m_x1 = point_estimate_M(x1, n)
    pe_m_x2 = point_estimate_M(x2, n)

    pe_d_x1 = point_estimate_D(x1, n, pe_m_x1)
    pe_d_x2 = point_estimate_D(x2, n, pe_m_x2)

    print('\nточечная оценка матожидания для x1:', pe_m_x1)
    intervals_for_M(n, pe_d_x1, pe_m_x1)
    print('\nточечная оценка матожидания для x2:', pe_m_x2)
    intervals_for_M(n, pe_d_x1, pe_m_x2)

    print('\nточечная оценка дисперсии для x1:', pe_d_x1)
    print('точечная оценка дисперсии для x2:', pe_d_x2)


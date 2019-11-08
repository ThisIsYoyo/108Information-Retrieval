from pathlib import Path

import numpy as np

from util import ListLoader

DATA = Path(__file__).parent / 'Data'
QUERY = DATA / 'Query'

term_list_loader = ListLoader(DATA, 'term_list.txt')
term_list = term_list_loader.get_list()

doc_term_freq_matrix = np.load('doc_term_freq_matrix.npy')
all_doc_len = np.sum(doc_term_freq_matrix, axis=0, dtype=np.int)  # (1, N)

random_matrix = np.load('random_matrix.npz')
p_w_given_T = random_matrix['p_w_given_T']
p_T_given_d = random_matrix['p_T_given_d']
T = 20  # num of Topics
N = 2265  # num of docs
W = len(term_list)  # num of terms
LOOP = 50


if __name__ == '__main__':

    for l in range(LOOP):
        p_w_given_T_numerator = np.zeros((W, T), dtype=np.float)
        p_t_sum = np.zeros(T, dtype=np.float)
        new_p_T_given_d = np.zeros((T, N), dtype=np.float)

        for j in range(N):
            # E step
            p_w_T = p_w_given_T * p_T_given_d[:, j]  # fixed doc i
            T_sum_given_w_col = np.sum(p_w_T, axis=1).reshape(W, 1)
            normal_p_w_T = p_w_T / T_sum_given_w_col

            p_w_T, T_sum_given_w_col = None, None  # release memory

            # M step
            weight_w_T = normal_p_w_T * doc_term_freq_matrix[:, j].reshape((W, 1))
            weight_w_sum_T = np.sum(weight_w_T, axis=0)  # (T, 1)
            p_w_given_T_numerator += weight_w_T
            p_t_sum += weight_w_sum_T

            new_p_T_given_d[:, j] = weight_w_sum_T / all_doc_len[j]

            weight_w_T, weight_w_sum_T = None, None  # release memory

        p_w_given_T = p_w_given_T_numerator / p_t_sum
        p_T_given_d = new_p_T_given_d

    np.savez(file='trained_matrix.npz', p_w_given_T=p_w_given_T, p_T_given_d=p_T_given_d)


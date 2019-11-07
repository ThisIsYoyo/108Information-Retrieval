import json
from pathlib import Path

import numpy as np

from util import ListLoader

DATA = Path(__file__).parent / 'Data'
QUERY = DATA / 'Query'

with open(DATA / 'doc_information.json', 'r') as fp:
    doc_information = json.load(fp)

term_list_loader = ListLoader(DATA, 'term_list.txt')
term_list = term_list_loader.get_list()

doc_term_freq_matrix = np.load('doc_term_freq_matrix.npy')
random_matrix = np.load('random_matrix.npz')
p_w_given_T = random_matrix['p_w_given_T']
p_T_given_d = random_matrix['p_T_given_d']
T = 15  # num of Topics
N = 2265  # num of docs
W = len(term_list)  # num of terms
LOOP = 10


if __name__ == '__main__':
    for l in range(LOOP):
        # E step
        p_T_given_w_d = np.zeros((W, N, T))  # (word_index, doc_index, topic_index)
        for i in range(N):
            t = p_T_given_d[:, i]  # fixed doc i
            p_w_T = p_w_given_T * t
            t_sum_given_w_row = np.sum(p_w_T, axis=1)
            t_sum_given_w_col = t_sum_given_w_row.reshape(W, 1)
            normal_p_w_T = p_w_T / t_sum_given_w_col

            p_T_given_w_d[:, i, :] = normal_p_w_T

            del t, p_w_T, t_sum_given_w_row, t_sum_given_w_col, normal_p_w_T

        # M step
        new_p_w_given_T = np.zeros((W, T))
        new_p_T_given_d = np.zeros((T, N))
        all_doc_len = np.sum(doc_term_freq_matrix, axis=0)  # (1, N)
        for k in range(T):  # fixed topic
            weight_w_d = doc_term_freq_matrix * p_T_given_w_d[:, :, k]

            weight_w_d_sum = np.sum(weight_w_d, axis=1)  # (W, 1)
            weight_w_d_all_sum = np.sum(weight_w_d)
            weight_w_d_sum_normal = weight_w_d_sum / weight_w_d_all_sum
            new_p_w_given_T[:, k] = weight_w_d_sum_normal

            weight_w_sum_d = np.sum(weight_w_d, axis=0)  # (1, N)
            weight_w_sum_d_normal = weight_w_sum_d / all_doc_len
            new_p_T_given_d[k, :] = weight_w_sum_d_normal

            del (weight_w_d, weight_w_d_sum, weight_w_d_all_sum, weight_w_d_sum_normal,
                 weight_w_sum_d, weight_w_sum_d_normal)

        p_w_given_T = new_p_w_given_T
        p_T_given_d = new_p_T_given_d

    np.savez(file='trained_matrix.npz', p_w_given_T=p_w_given_T, p_T_given_d=p_T_given_d)


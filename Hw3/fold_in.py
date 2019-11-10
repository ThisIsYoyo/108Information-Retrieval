import time

import numpy as np

T = 128
N = 2265
LOOP = 10

if __name__ == '__main__':
    # random fold in random matrix
    # p_t_d_tmp = np.random.rand(T, N)
    # p_t_d = p_t_d_tmp / np.sum(p_t_d_tmp, axis=0)
    # p_t_d_tmp = None
    p_t_d = np.load('fold_in_p_t_d.npy')

    p_w_t = np.load('p_w_given_T.npy')  # W x T
    W = p_w_t.shape[0]

    doc_term_freq_matrix = np.load('doc_term_freq_matrix.npy')  # W x N
    all_doc_len = np.sum(doc_term_freq_matrix, axis=0)  # N x 1

    for l in range(LOOP):
        start = time.time()
        new_p_t_d = np.zeros((T, N))
        for m in range(N):
            # E step
            p_w = p_w_t * p_t_d[:, m]  # p_w_t fixed doc_m
            p_w_t_sum = np.sum(p_w, axis=1).reshape(W, 1)
            p_w_given_T = p_w / p_w_t_sum
            p_w, p_w_t_sum = None, None

            # M step
            weight_p_w_t = doc_term_freq_matrix[:, m].reshape(W, 1) * p_w_given_T
            weight_p_w_sum_t = np.sum(weight_p_w_t, axis=0)
            new_p_t_d[:, m] = weight_p_w_sum_t / all_doc_len[m]
            weight_p_w_t, weight_p_w_sum_t = None, None
        p_t_d = new_p_t_d

        end = time.time()
        print(f'{l}-th train spend {end - start} secs')
        np.save('fold_in_p_t_d.npy', p_t_d)




import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Custom Var
T = 128
LOOP = 2

# Fixed var
COLLECTION_PATH = Path(__file__).parent / 'Data' / 'Collection.txt'


if __name__ == '__main__':
    start_time = time.time()

    # deal collection
    with open(COLLECTION_PATH, 'r') as fp:
        collection_line_list = fp.readlines()
    cs = CountVectorizer(dtype=np.int)
    doc_term_freq_matrix = cs.fit_transform(collection_line_list)
    doc_term_freq_matrix = np.transpose(doc_term_freq_matrix)

    term_len, doc_len = doc_term_freq_matrix.shape
    term_list = cs.get_feature_names()
    all_doc_len = np.sum(doc_term_freq_matrix, axis=0)

    collection_line_list, cs = None, None

    coll_end_time = time.time()
    print(f'deal collection finish with {coll_end_time-start_time} secs')

    # initial random matrix
    p_w_given_T_tmp = np.random.rand(term_len, T)
    p_T_given_d_tmp = np.random.rand(T, doc_len)
    p_w_given_T = p_w_given_T_tmp / np.sum(p_w_given_T_tmp, axis=0)
    p_T_given_d = p_T_given_d_tmp / np.sum(p_T_given_d_tmp, axis=0)

    p_w_given_T_tmp, p_T_given_d_tmp = None, None

    initial_time = time.time()
    print(f'initial random matrix finish with {initial_time - coll_end_time} secs')

    # train EM model
    for l in range(LOOP):
        start_train_time = time.time()

        p_w_given_T_numerator = np.zeros((term_len, T), dtype=np.float)
        p_t_sum = np.zeros(T, dtype=np.float)
        new_p_T_given_d = np.zeros((T, doc_len), dtype=np.float)

        for j in range(doc_len):
            # E step
            p_w_T = p_w_given_T * p_T_given_d[:, j]  # fixed doc j
            T_sum_given_w_col = np.sum(p_w_T, axis=1).reshape(term_len, 1)
            normal_p_w_T = p_w_T / T_sum_given_w_col

            p_w_T, T_sum_given_w_col = None, None  # release memory

            # M step
            weight_w_T = normal_p_w_T * doc_term_freq_matrix[:, j].reshape((term_len, 1)).toarray()
            weight_w_sum_T = np.sum(weight_w_T, axis=0)  # (T, 1)
            p_w_given_T_numerator += weight_w_T
            p_t_sum += weight_w_sum_T

            new_p_T_given_d[:, j] = weight_w_sum_T / all_doc_len[0, j]

            weight_w_T, weight_w_sum_T = None, None  # release memory

        p_w_given_T = p_w_given_T_numerator / p_t_sum
        p_T_given_d = new_p_T_given_d

        end_train_time = time.time()
        print(f'{j} train finish with {end_train_time - start_train_time} secs')









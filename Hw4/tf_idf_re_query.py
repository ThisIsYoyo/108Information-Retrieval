from pathlib import Path

import numpy as np

from util import ListLoader, TermLoader

STATS = Path(__file__).parent / 'Stats_Data'
DATA = Path(__file__).parent / 'Data'
N = 2265
a = 0.3
b = 0.7

doc_term_freq_matrix = np.load(STATS / 'doc_term_frequency.npy')
bool_matrix = doc_term_freq_matrix.astype(bool)
exist_matrix = bool_matrix.astype(float)

term_appearance_matrix = np.sum(exist_matrix, axis=1)  # (1, W)
idf_matrix = np.log((term_appearance_matrix ** -1) * N)


if __name__ == '__main__':
    query_list_loader = ListLoader(path=DATA, file_name='query_list.txt')
    query_list = query_list_loader.get_list()

    doc_list_loader = ListLoader(path=DATA, file_name='doc_list.txt')
    doc_list = doc_list_loader.get_list()

    term_list_loader = ListLoader(path=STATS, file_name='doc_term_list.txt')
    term_list = term_list_loader.get_list()
    term_len = len(term_list)

    with open(STATS / 'plsa_ranking_top_15.txt', 'r') as fp:
        plsa_top_15_list = fp.readlines()

    re_query_matrix = np.zeros((term_len, len(query_list)))
    for query_index, query in enumerate(query_list):
        q_term_loader = TermLoader(path=DATA / 'Query', file_name=query)

        q_term_freq = np.zeros(term_len)
        for term in q_term_loader.iter_term():
            if term not in term_list:
                continue

            term_index = term_list.index(term)
            q_term_freq[term_index] += 1

            term_index = None

        q_score = q_term_freq * idf_matrix

        plsa_top_15 = plsa_top_15_list[query_index].split()
        doc_score_sum = np.zeros(term_len)
        for plsa_result in plsa_top_15:
            doc_index = doc_list.index(plsa_result)
            doc_freq = doc_term_freq_matrix[:, doc_index]
            doc_score = doc_freq * idf_matrix
            doc_score_sum += doc_score

            doc_index, doc_freq, doc_score = None, None, None

        re_query_matrix[:, query_index] = a * q_score + b * (len(plsa_top_15) ** -1) * doc_score_sum

    np.save(STATS / 're_query_matrix.npy', re_query_matrix)


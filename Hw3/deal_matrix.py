from pathlib import Path

import numpy

from util import ListLoader, TermLoader

DATA = Path(__file__).parent / 'Data'
T = 15  # topic amount


if __name__ == '__main__':
    doc_list_loader = ListLoader(path=DATA, file_name='doc_list.txt')
    doc_list = doc_list_loader.get_list()

    term_set = set()
    doc_information = {}
    for doc in doc_list:
        doc_term_loader = TermLoader(path=DATA / 'Document', file_name=doc)
        doc_term_loader.set_start_line(start_line=3)

        doc_term_frequency = {}
        for term in doc_term_loader.iter_term():
            doc_term_frequency.setdefault(term, 0)
            doc_term_frequency[term] += 1
            term_set.add(term)

        doc_information[doc] = doc_term_frequency

    sorted_term_list = sorted([int(i) for i in term_set])  # int
    with open(DATA / 'term_list.txt', 'w') as fp:
        for sorted_term in sorted_term_list:
            fp.write(f'{sorted_term}\n')

    doc_term_freq_matrix = numpy.zeros((len(sorted_term_list), len(doc_list)))
    for doc_index, doc in enumerate(doc_list):
        for term, freq in doc_information[doc].items():
            term_index = sorted_term_list.index(int(term))
            doc_term_freq_matrix[term_index, doc_index] = freq

    term_list_len = len(sorted_term_list)
    doc_list_len = len(doc_list)  # 2265
    w_given_T = numpy.random.rand(term_list_len, T)
    T_given_d = numpy.random.rand(T, doc_list_len)
    w_given_T_col_sum = numpy.sum(w_given_T, axis=0)
    T_given_d_col_sum = numpy.sum(T_given_d, axis=0)
    p_w_given_T = w_given_T / w_given_T_col_sum
    p_T_given_d = T_given_d / T_given_d_col_sum

    numpy.save('doc_term_freq_matrix.npy', doc_term_freq_matrix)
    numpy.savez('random_matrix.npz', p_w_given_T=p_w_given_T, p_T_given_d=p_T_given_d)

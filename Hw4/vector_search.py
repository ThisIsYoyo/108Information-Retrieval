from pathlib import Path

import numpy as np
from numpy import float16
from numpy.linalg import norm

from util import FreqMatrixTransformer

DATA = Path(__file__).parent / 'Data'
STATS = Path(__file__).parent / 'Stats_Data'
N = 2265  # Number of doc
RELATE_AMOUNT = 12
a = 1
b = 0.7
c = 0.15

if __name__ == '__main__':
    # vectorize doc
    doc_transformer = FreqMatrixTransformer()
    doc_transformer.set_x_axis(x_flie=DATA / 'doc_list.txt')
    doc_transformer.set_doc_content_start_line(3)

    doc_term_freq_matrix = doc_transformer.vectorize(folder=DATA / 'Document')
    doc_list = doc_transformer.x_axis
    term_list = doc_transformer.y_axis
    term_len = len(term_list)

    doc_transformer = None

    # produce term_idf
    doc_term_exist = doc_term_freq_matrix.astype(bool)
    doc_term_appear = doc_term_exist.astype(float16)
    term_appearence = np.sum(doc_term_appear, axis=1)
    term_idf = np.log((term_appearence ** -1) * N + 1)

    doc_term_exist, term_appearence = None, None

    # vectorize query
    query_transformer = FreqMatrixTransformer()
    query_transformer.set_x_axis(x_flie=DATA / 'query_list.txt')
    query_transformer.set_y_axis(y_list=term_list)

    query_term_freq_matrix = query_transformer.vectorize(DATA / 'Query')
    query_term_exist = query_term_freq_matrix.astype(bool)
    query_term_appear = query_term_exist.astype(float16)
    query_list = query_transformer.x_axis
    query_len = len(query_list)

    query_transformer, query_term_exist = None, None

    # caculate tf or idf
    query_idf = query_term_appear * term_idf.reshape(term_len, 1)
    doc_idf = doc_term_appear * term_idf.reshape(term_len, 1)
    doc_tf = doc_term_freq_matrix + 1

    # np.save(STATS / 'query_tf_idf.npy', query_idf)
    # np.save(STATS / 'doc_tf_idf.npy', doc_tf)

    # loading data
    # query_tf_idf = np.load(STATS / 'query_tf_idf.npy')
    # doc_tf_idf = np.load(STATS / 'doc_tf_idf.npy')
    #
    # with open(DATA / 'doc_list.txt', 'r') as fp:
    #     doc_list = [item.strip() for item in fp.readlines()]
    #
    # with open(DATA / 'query_list.txt', 'r') as fp:
    #     query_list = [item.strip() for item in fp.readlines()]
    # query_len = len(query_list)
    #
    # with open(STATS / 'term_list.txt') as fp:
    #     term_list = [item.strip() for item in fp.readlines()]
    # term_len = len(term_list)

    # first query
    query_relate_doc_index = np.zeros((query_len, RELATE_AMOUNT), dtype=np.int)
    for query_index in range(query_len):
        doc_score_molecule = query_idf[:, query_index].dot(doc_tf)
        doc_score_denominator = norm(query_idf[:, query_index]) * norm(doc_tf, axis=0)
        doc_score = doc_score_molecule / doc_score_denominator

        doc_score_with_index = [(score, doc_index) for doc_index, score in enumerate(doc_score)]
        doc_score_with_index = sorted(doc_score_with_index, reverse=True)

        for i in range(RELATE_AMOUNT):
            query_relate_doc_index[query_index, i] = doc_score_with_index[i][1]
        print(f'finish {query_index}-th query search')

    # np.save(STATS / 'query_top_relate_index.npy', query_relate_doc_index)

    # re vectorize query
    # query_relate_doc_index = np.load(STATS / 'query_top_relate_index.npy')

    doc_set = set(doc_list)
    re_query_idf = np.zeros((term_len, query_len))
    for query_index in range(query_len):

        rela_doc_set = set()
        rela_doc_idf_sum = np.zeros(term_len)
        for top_doc_index in query_relate_doc_index[query_index]:
            rela_doc_idf_sum += doc_idf[:, top_doc_index]
            rela_doc_set.add(doc_list[top_doc_index])

        non_rela_doc_idf_sum = np.zeros(term_len)
        for non_rela_doc in doc_set - rela_doc_set:
            doc_index = doc_list.index(non_rela_doc)
            non_rela_doc_idf_sum += doc_idf[:, doc_index]

        re_query_idf[:, query_index] = a * query_idf[:, query_index] + b / RELATE_AMOUNT * rela_doc_idf_sum - \
                                       c / (N - RELATE_AMOUNT) * non_rela_doc_idf_sum

    # second query

    result = open(Path(__file__).parent / 'submission.csv', 'w')
    result.write('Query,RetrievedDocuments')

    for query_index in range(query_len):
        result.write(f'\n{query_list[query_index]},')

        doc_score_molecule = re_query_idf[:, query_index].dot(doc_tf)
        doc_score_denominator = norm(re_query_idf[:, query_index]) * norm(doc_tf, axis=0)
        doc_score = doc_score_molecule / doc_score_denominator

        doc_score_with_index = [(doc_score[doc_index], doc_index) for doc_index in range(N)]
        doc_score_with_index = sorted(doc_score_with_index, reverse=True)

        for i in range(50):
            doc_index = doc_score_with_index[i][1]
            result.write(f'{doc_list[doc_index]} ')

        print(f'finish {query_index}-th requery search')

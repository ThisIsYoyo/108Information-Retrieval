from pathlib import Path

import numpy as np

from util import BGLMLoader, ListLoader, TermLoader

DATA = Path(__file__).parent / 'Data'
STATS = Path(__file__).parent / 'Stats_Data'

a = 0.33
b = 0.34
N = 2265

doc_term_freq_matrix = np.load('doc_term_freq_matrix.npy')  # w x d
p_w_d = doc_term_freq_matrix / np.sum(doc_term_freq_matrix, axis=0)  # w x d
p_w_d = np.nan_to_num(p_w_d)

p_w_given_T = np.load('p_w_given_T.npy')
p_T_given_d = np.load('p_T_given_d.npy')
plsa_p_w_d = p_w_given_T.dot(p_T_given_d)  # w x d

bglm_loader = BGLMLoader(path=DATA, file_name='BGLM.txt')
bglm_loader.parse()
bglm_array = np.array(bglm_loader.get_bglm_list())  # w x 1 (log P_BG)

if __name__ == '__main__':
    query_list_loader = ListLoader(path=DATA, file_name='query_list.txt')
    query_list = query_list_loader.get_list()

    term_list_loader = ListLoader(path=STATS, file_name='doc_term_list.txt')
    term_list = term_list_loader.get_list()

    doc_list_loader = ListLoader(path=DATA, file_name='doc_list.txt')
    doc_list = doc_list_loader.get_list()

    plsa_ranking_top_15 = open('plsa_ranking_top_15.txt', 'w')

    for query in query_list:
        term_loader = TermLoader(path=DATA / 'Query', file_name=query)

        doc_ranking = np.zeros(N)
        for term in term_loader.iter_term():
            term_int = int(term)
            term_index = term_list.index(term) if term in term_list else -1

            if term_index != -1:
                part_1 = np.log(a) + np.log(p_w_d[term_index, :])
                part_2 = np.log(b) + np.log(plsa_p_w_d[term_index, :])
                part_3 = np.tile(np.log(1 - a - b) + bglm_array[term_int], 2265)
                part_1 = np.nan_to_num(part_1)

                term_doc_ranking = np.logaddexp(part_1, part_2, part_3)
                doc_ranking += term_doc_ranking

                del part_1, part_2, part_3
            else:
                term_doc_ranking = np.tile(np.log(1 - a - b) + bglm_array[term_int], 2265)
                doc_ranking += term_doc_ranking

            del term_doc_ranking

        ranking_with_doc = [(doc_ranking[i], doc_list[i]) for i in range(2265)]
        ranking_with_doc = sorted(ranking_with_doc, reverse=True)

        for ranking_doc in ranking_with_doc[:15]:
            plsa_ranking_top_15.write(f'{ranking_doc[1]} ')
        plsa_ranking_top_15.write('\n')

    plsa_ranking_top_15.close()


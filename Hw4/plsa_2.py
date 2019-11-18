from pathlib import Path

import numpy as np

from util import BGLMLoader, ListLoader, TermLoader

DATA = Path(__file__).parent / 'Data'
STATS = Path(__file__).parent / 'Stats_Data'

a = 0.3
b = 0.4
N = 2265

doc_term_freq_matrix = np.load(STATS / 'doc_term_frequency.npy')  # w x d
p_w_d = doc_term_freq_matrix / np.sum(doc_term_freq_matrix, axis=0)  # w x d

p_w_given_T = np.load(STATS / 'p_w_given_T.npy')
p_T_given_d = np.load(STATS / 'p_T_given_d.npy')
plsa_p_w_d = p_w_given_T.dot(p_T_given_d)  # w x d

bglm_loader = BGLMLoader(path=DATA, file_name='BGLM.txt')
bglm_loader.parse()
bglm_array = np.array(bglm_loader.get_bglm_list())  # w x 1 (log P_BG)

if __name__ == '__main__':
    term_list_loader = ListLoader(path=STATS, file_name='doc_term_list.txt')
    term_list = term_list_loader.get_list()
    term_bglm_list = [bglm_array[int(term)] for term in term_list]
    term_bglm_array = np.array(term_bglm_list)

    del term_bglm_list

    all_part_1 = np.log(a) + np.log(p_w_d)
    all_part_1 = np.nan_to_num(all_part_1)
    all_part_2 = np.log(b) + np.log(plsa_p_w_d)
    all_part_3 = np.tile((np.log(1 - a - b) + term_bglm_array), N)
    caculate_p_w_d = np.logaddexp(all_part_1, all_part_2, all_part_3)

    del all_part_1, all_part_2, all_part_3

    re_query_matrix = np.load(STATS / 're_query_matrix.npy')

    query_list_loader = ListLoader(path=DATA, file_name='query_list.txt')
    query_list = query_list_loader.get_list()

    doc_list_loader = ListLoader(path=DATA, file_name='doc_list.txt')
    doc_list = doc_list_loader.get_list()

    submission = open('submission.csv', 'w')
    submission.write('Query,RetrievedDocuments')

    for query_index, query in enumerate(query_list):
        submission.write(f'\n{query},')

        doc_ranking = list(re_query_matrix[:, query_index].dot(caculate_p_w_d))

        ranking_with_doc = [(doc_ranking[i], doc_list[i]) for i in range(N)]
        ranking_with_doc = sorted(ranking_with_doc, reverse=True)

        for ranking_doc in ranking_with_doc[:50]:
            submission.write(f'{ranking_doc[1]} ')

    submission.close()

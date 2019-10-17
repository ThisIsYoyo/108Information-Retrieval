import json
from math import log
from pathlib import Path
from typing import Dict

from util import ListLoader, TermLoader

K1 = 1.85
K3 = 1.85
B = 0.75
N = 2265
DATA_PATH = Path(__file__).parent / 'Data'
QUERY_PATH = Path(__file__).parent / 'Data' / 'Query'
DOCUMENT_PATH = Path(__file__).parent / 'Data' / 'Document'

with open(DATA_PATH / 'global_information.json', 'r') as fp:
    global_information = json.loads(fp.read())
AVE_DOC_LEN = global_information['average_doc_length']

with open(DATA_PATH / 'doc_information.json', 'r') as fp:
    doc_information = json.loads(fp.read())


def sim(doc_freq: [int], doc_len: int, q_freq: [int], name_space: [str]) -> float:
    doc_len_normal = K1 * ((1 - B) + (B * doc_len / AVE_DOC_LEN))

    # td and idf
    sim_result = 0
    for i in range(len(doc_freq)):
        term = name_space[i]

        fun_doc = (K1 + 1) * doc_freq[i] / (doc_len_normal + doc_freq[i])
        fun_q = (K3 + 1) * q_freq[i] / (K3 + q_freq[i])
        idf = log((N - global_information[term] + 0.5) / (global_information[term] + 0.5))

        sim_result += fun_doc * fun_q * idf

    return sim_result


def ranking_doc_from(rank_dict: Dict):  # rank_dict = doc: rank
    rank_tmp_list = [(rank, doc) for doc, rank in rank_dict.items()]
    rank_list = [doc for rank, doc in sorted(rank_tmp_list, reverse=True)]

    return rank_list


if __name__ == '__main__':
    query_list_loader = ListLoader(path=DATA_PATH, file_name='query_list.txt')
    query_list = query_list_loader.get_list()

    submission = open(DATA_PATH / 'submission.csv', 'w')
    submission.write('Query,RetrievedDocuments\n')

    for query in query_list:
        query_loader = TermLoader(path=QUERY_PATH, file_name=query)

        query_term_frequency = {}
        query_len = 0
        query_name_space = set()
        for term in query_loader.iter_term():
            query_term_frequency.setdefault(term, 0)
            query_term_frequency[term] += 1
            query_len += 1
            query_name_space.add(term)

        ranking = {}  # doc: ranking
        # Caculate sim each doc
        for doc, information in doc_information.items():
            doc_len = information['length']
            doc_name_space = set(information['name_space'])
            inter_name_space = doc_name_space.intersection(query_name_space)

            query_frequency = [query_term_frequency[term] for term in inter_name_space]
            doc_frequency = [information[term] for term in inter_name_space]

            doc_rank = \
                sim(doc_freq=doc_frequency, doc_len=doc_len, q_freq=query_frequency, name_space=list(inter_name_space))
            ranking[doc] = doc_rank

        ranking_list = ranking_doc_from(rank_dict=ranking)

        submission.write(f'{query},')
        for doc in ranking_list:
            submission.write(f'{doc} ')
        submission.write('\n')






import json
from pathlib import Path

import numpy as np

from Hw1.pre_process import ListLoader

DATA_PATH = Path(__file__).parent / 'Data'
QUERY_PATH = Path(__file__).parent / 'Data' / 'Query'
N = 2265  # Number of doc

if __name__ == '__main__':
    query_list_loader = ListLoader(path=DATA_PATH, file_name='query_list.txt')
    query_list = query_list_loader.get_list()

    with open(DATA_PATH / 'doc_term_frequency.json', 'r') as fp:
        doc_term_frequency = json.loads(fp.read())  # doc: {term: frequency}

    with open(DATA_PATH / 'term_idf.json', 'r') as fp:
        term_idf = json.loads(fp.read())  # term: frequency

    result = open(Path(__file__).parent / 'submission.csv', 'w')
    result.write('Query,RetrievedDocuments\n')

    # Start searching
    for query in query_list:
        query_loader = ListLoader(path=QUERY_PATH, file_name=query)
        query_term_list = query_loader.get_list()

        # Sort query {term: frequency}
        query_term_frequency_dict = {}
        for query_term in query_term_list:
            query_term_frequency_dict.setdefault(query_term, 0)
            query_term_frequency_dict[query_term] += 1

        # Separate query term and query frequency into two matrix
        term_list = []
        query_frequency_list = []

        for term, frequency in query_term_frequency_dict.items():
            term_list.append(term)
            query_frequency_list.append(frequency)

        # # Pre-process query frequency
        # query_weight_frequency_list = []
        # for frequency in query_frequency_list:
        #     new_frequency = 0.5 + 0.5 * frequency / max(query_frequency_list)
        #     query_weight_frequency_list.append(new_frequency)

        # Caculate sim of every doc
        doc_rank_dict = {}
        for doc, term_frequency in doc_term_frequency.items():
            all_term_in_doc = [k for k, v in term_frequency.items()]
            all_term_freq_in_doc = [v for k, v in term_frequency.items()]

            #  Caculate weight of doc
            doc_weight = []
            query_weight = []
            for i in range(len(all_term_in_doc)):
                term = all_term_in_doc[i]
                frequency = all_term_freq_in_doc[i]

                weight = (1 + frequency) * term_idf[term]
                doc_weight.append(weight)

                if term in term_list:
                    weight = (1 + query_term_frequency_dict[term]) * term_idf[term]
                    query_weight.append(weight)
                else:
                    query_weight.append(0)

            # # Caculate doc term frequency contract to term in query -> just for dot
            # doc_frequency_list = []  # len(doc_frequency_list) == len(query_frequency_list)
            # for term in term_list:
            #     if term in term_frequency.keys():
            #         doc_frequency_list.append(term_frequency[term])
            #     else:
            #         doc_frequency_list.append(0)
            #
            # # Caculate idf that query term own
            # idf_list = []
            # for i in range(len(doc_frequency_list)):
            #     term = term_list[i]
            #
            #     if term in global_term_appearance.keys():
            #         idf = log(N / global_term_appearance[term]) if global_term_appearance[term] != 0 else 0
            #     else:
            #         idf = 0
            #     idf_list.append(idf)

            # doc_weight_vector = np.array(doc_frequency_list) * np.array(idf_list)
            # query_weight_vector = np.array(query_weight_frequency_list) * np.array(idf_list)
            norm_of_doc_weight = np.linalg.norm(doc_weight)
            norm_of_query_weight = np.linalg.norm(query_weight)
            sim_doc_query = np.dot(doc_weight, query_weight) / (norm_of_doc_weight * norm_of_query_weight)

            doc_rank_dict[doc] = sim_doc_query

        query_search_result = sorted([(v, k) for k, v in doc_rank_dict.items()], reverse=True)

        result.write(f'{query},')
        for doc_sim, doc in query_search_result:
            result.write(f'{doc} ')
        result.write('\n')


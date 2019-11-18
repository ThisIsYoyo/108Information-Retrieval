import json
from pathlib import Path

import numpy as np

from Hw1.pre_process import ListLoader, TermLoader

DATA_PATH = Path(__file__).parent / 'Data'
QUERY_PATH = Path(__file__).parent / 'Data' / 'Query'
N = 2265  # Number of doc

if __name__ == '__main__':
    query_list_loader = ListLoader(path=DATA_PATH, file_name='query_list.txt')
    query_list = query_list_loader.get_list()

    with open(DATA_PATH / 'doc_term_frequency.json', 'r') as fp:
        all_doc_term_frequency = json.loads(fp.read())  # doc: {term: frequency}

    with open(DATA_PATH / 'global_term_appearance.json', 'r') as fp:
        global_term_appearance = json.loads(fp.read())  # term: frequency

    result = open(Path(__file__).parent / 'submission.csv', 'w')
    result.write('Query,RetrievedDocuments\n')

    # Start searching
    for query in query_list:
        query_loader = TermLoader(path=QUERY_PATH, file_name=query)

        # Sort query {term: frequency}
        query_term_frequency = {}
        for query_term in query_loader.iter_term():
            query_term_frequency.setdefault(query_term, 0)
            query_term_frequency[query_term] += 1

        # Caculate sim of every doc
        doc_rank_dict = {}
        for doc, doc_term_frequency in all_doc_term_frequency.items():
            # make vector name space
            query_term_space = set(k for k, v in query_term_frequency.items())
            doc_term_space = set(k for k, v in doc_term_frequency.items())
            term_space = query_term_space.union(doc_term_space)

            query_frequency_vector = []
            doc_frequency_vector = []
            idf_vector = []
            for term in term_space:
                # query_vector
                if term in query_term_frequency:
                    frequency = query_term_frequency[term]
                    query_frequency_vector.append(frequency)
                else:
                    query_frequency_vector.append(0)

                # doc_vector
                if term in doc_term_frequency:
                    frequency = doc_term_frequency[term]
                    doc_frequency_vector.append(frequency)
                else:
                    doc_frequency_vector.append(0)

                # idf_vector
                if term in global_term_appearance:
                    frequency = global_term_appearance[term]
                    idf = np.log(N / frequency)
                    idf_vector.append(idf)
                else:
                    idf_vector.append(0)

            query_weight_vector = np.multiply(query_frequency_vector, idf_vector)
            doc_weight_vector = np.multiply(doc_frequency_vector, idf_vector)
            dot_result = np.dot(query_weight_vector, doc_weight_vector)

            norm_of_query_vector = np.linalg.norm(query_weight_vector)
            norm_of_doc_vector = np.linalg.norm(doc_weight_vector)

            rank = dot_result / (norm_of_query_vector * norm_of_doc_vector)
            doc_rank_dict[doc] = rank

        query_search_result = sorted([(v, k) for k, v in doc_rank_dict.items()], reverse=True)

        result.write(f'{query},')
        for doc_sim, doc in query_search_result:
            result.write(f'{doc} ')
        result.write('\n')

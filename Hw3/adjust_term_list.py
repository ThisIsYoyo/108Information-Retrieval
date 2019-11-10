from pathlib import Path

# Fixed var
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from util import ListLoader, TermLoader

DATA = Path(__file__).parent / 'Data'
COLLECTION_PATH = DATA / 'Collection.txt'


if __name__ == '__main__':

    # deal collection
    # with open(COLLECTION_PATH, 'r') as fp:
    #     collection_line_list = fp.readlines()
    # cs = CountVectorizer(dtype=np.int)
    # collection_doc_term_freq_matrix = cs.fit_transform(collection_line_list)
    # coll_term_list = list(cs.get_feature_names())
    #
    # doc_list_loader = ListLoader(path=DATA, file_name='doc_list.txt')
    # doc_list = doc_list_loader.get_list()
    # doc_line_list = []
    # for doc in doc_list:
    #     term_loader = TermLoader(path=DATA / 'Document', file_name=doc)
    #     term_list = list(term_loader.iter_term())
    #     term_line = ' '.join(term_list)
    #     doc_line_list.append(term_line)
    #
    # cs_doc = CountVectorizer(dtype=np.int)
    # doc_term_freq_matrix = cs_doc.fit_transform(doc_line_list)
    # term_list = cs_doc.get_feature_names()
    #
    # np.save('coll_term_list.npy', np.array(coll_term_list))
    # np.save('term_list.npy', np.array(term_list))

    coll_term_list = list(np.load('coll_term_list.npy'))

    doc_list_loader = ListLoader(path=DATA, file_name='doc_list.txt')
    doc_list = doc_list_loader.get_list()
    doc_term_freq_matrix = np.zeros((len(coll_term_list), len(doc_list)), dtype=np.int)
    for doc_index, doc in enumerate(doc_list):

        term_loader = TermLoader(path=DATA / 'Document', file_name=doc)
        term_loader.set_start_line(3)
        for term in term_loader.iter_term():
            if term not in coll_term_list:
                continue

            term_index = coll_term_list.index(term)
            doc_term_freq_matrix[term_index, doc_index] += 1
        pass

    np.save('doc_term_freq_matrix.npy', doc_term_freq_matrix)




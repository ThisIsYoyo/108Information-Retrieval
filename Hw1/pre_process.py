import abc
import json
from pathlib import Path

from util import ListLoader, TermLoader

DATA_PATH = Path(__file__).parent / 'Data'
DOCUMENT_PATH = Path(__file__).parent / 'Data' / 'Document'


if __name__ == '__main__':
    doc_list_loader = ListLoader(path=DATA_PATH, file_name='doc_list.txt')
    doc_list = doc_list_loader.get_list()

    global_term_appearance = {}  # term: appear times in docs
    doc_term_frequency_dict = {}  # doc: {term: appear frequency in doc}
    for doc in doc_list:
        doc_loader = TermLoader(path=DOCUMENT_PATH, file_name=doc)
        doc_loader.set_start_line(3)

        term_frequency = {}
        for term in doc_loader.iter_term():
            term_frequency.setdefault(term, 0)
            term_frequency[term] += 1

        # doc_loader = ListLoader(path=DOCUMENT_PATH, file_name=doc)
        # doc_loader.set_start_line(start_line=3)
        # doc_term_list = doc_loader.get_list()
        #
        # term_frequency = {}
        # for term in doc_term_list:
        #     term_frequency.setdefault(term, 0)
        #     term_frequency[term] += 1

        for k, v in term_frequency.items():
            global_term_appearance.setdefault(k, 0)
            global_term_appearance[k] += 1

        doc_term_frequency_dict[doc] = term_frequency

    with open(str(DATA_PATH / 'doc_term_frequency.json'), 'w') as fp:
        json.dump(doc_term_frequency_dict, fp, indent=4)

    # term_idf = {}
    # for k, v in global_term_appearance.items():
    #     term_idf[k] = log(N / v)
    with open(DATA_PATH / 'term_idf.json', 'w') as fp:
        json.dump(global_term_appearance, fp, indent=4)

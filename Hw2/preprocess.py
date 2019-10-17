import json
from pathlib import Path

from util import ListLoader, TermLoader

DATA_PATH = Path(__file__).parent / 'Data'
DOCUMENT_PATH = Path(__file__).parent / 'Data' / 'Document'


if __name__ == '__main__':
    doc_name_list_loader = ListLoader(path=DATA_PATH, file_name='doc_list.txt')
    doc_name_list = doc_name_list_loader.get_list()

    global_information_dict = {}  # term1: appearence, term2: appearence, ... , average_doc_length: average_doc_length
    doc_information_dict = {}  # doc: {term1: frequency, term2: frequency, ... , length: doc_len}
    sum_of_doc_len = 0
    for doc_name in doc_name_list:
        loader = TermLoader(path=DOCUMENT_PATH, file_name=doc_name)
        loader.set_start_line(3)

        length = 0
        term_frequency_dict = {}  # term: frequency
        name_space = set()
        for term in loader.iter_term():
            term_frequency_dict.setdefault(term, 0)
            term_frequency_dict[term] += 1
            length += 1
            name_space.add(term)

        doc_information_dict[doc_name] = term_frequency_dict
        doc_information_dict[doc_name]['length'] = length
        doc_information_dict[doc_name]['name_space'] = list(name_space)
        sum_of_doc_len += length

        for term, frequency in term_frequency_dict.items():
            global_information_dict.setdefault(term, 0)
            global_information_dict[term] += 1

    global_information_dict['average_doc_length'] = sum_of_doc_len / len(doc_name_list)

    with open(DATA_PATH / 'doc_information.json', 'w') as fp:
        json.dump(doc_information_dict, fp, indent=4)

    with open(DATA_PATH / 'global_information.json', 'w') as fp:
        json.dump(global_information_dict, fp, indent=4)








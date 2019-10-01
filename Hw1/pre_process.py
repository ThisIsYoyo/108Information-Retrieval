import abc
import json
from pathlib import Path

from numpy.ma import log

DATA_PATH = Path(__file__).parent / 'Data'
DOCUMENT_PATH = Path(__file__).parent / 'Data' / 'Document'
N = 2265


class BaseLoader:
    def __init__(self, path: Path, file_name: str):
        self.real_file_path = path / file_name
        self.start_line = 0

    @abc.abstractmethod
    def read(self):
        pass


class ListLoader(BaseLoader):

    def read(self):
        with open(str(self.real_file_path), 'r') as fp:
            text = fp.read()
        return text

    def get_list(self):
        all_list = []
        for line in self._iter_line_list():
            list_in_line = line.split()
            list_in_line = list_in_line if len(list_in_line) == 1 else list_in_line[:-1]

            all_list.extend(list_in_line)
        return all_list

    def set_start_line(self, start_line: int):
        self.start_line = start_line

    def _iter_line_list(self) -> str:
        with open(str(self.real_file_path), 'r') as fp:
            line_list = fp.readlines()

        wanted_list = line_list[self.start_line:]
        for line in wanted_list:
            yield line.strip()


if __name__ == '__main__':
    doc_list_loader = ListLoader(path=DATA_PATH, file_name='doc_list.txt')
    doc_list = doc_list_loader.get_list()

    global_term_appearance = {}
    doc_term_frequency_dict = {}  # doc: {term: frequency}
    for doc in doc_list:
        doc_loader = ListLoader(path=DOCUMENT_PATH, file_name=doc)
        doc_loader.set_start_line(start_line=3)
        doc_term_list = doc_loader.get_list()

        term_frequency = {}
        for term in doc_term_list:
            term_frequency.setdefault(term, 0)
            term_frequency[term] += 1

        for k, v in term_frequency.items():
            global_term_appearance.setdefault(k, 0)
            global_term_appearance[k] += 1

        doc_term_frequency_dict[doc] = term_frequency

    with open(str(DATA_PATH / 'doc_term_frequency.json'), 'w') as fp:
        json.dump(doc_term_frequency_dict, fp, indent=4)

    term_idf = {}
    for k, v in global_term_appearance.items():
        term_idf[k] = log(N / v)
    with open(DATA_PATH / 'term_idf.json', 'w') as fp:
        json.dump(term_idf, fp, indent=4)

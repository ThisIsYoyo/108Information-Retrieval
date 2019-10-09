import abc
import json
from pathlib import Path

DATA_PATH = Path(__file__).parent / 'Data'
DOCUMENT_PATH = Path(__file__).parent / 'Data' / 'Document'
# N = 2265


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


class TermLoader(BaseLoader):

    def read(self):
        with open(str(self.real_file_path), 'r') as fp:
            text = fp.read()
        return text

    def iter_term(self, skip_last_token: bool = True):
        for line in self._iter_line_list():
            term_in_line = line.split()
            term_in_line = term_in_line[:-1] if skip_last_token else term_in_line
            for term in term_in_line:
                yield term

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

import abc
from pathlib import Path
from typing import List, Iterable


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
        try:
            for line in self._iter_line_list():
                term_in_line = line.split()
                term_in_line = term_in_line[:-1] if skip_last_token else term_in_line
                for term in term_in_line:
                    yield term
        except:
            yield ''

    def set_start_line(self, start_line: int):
        self.start_line = start_line

    def _iter_line_list(self) -> str:
        with open(str(self.real_file_path), 'r') as fp:
            line_list = fp.readlines()

        if len(line_list) < self.start_line + 1:
            return ''

        wanted_list = line_list[self.start_line:]
        for line in wanted_list:
            yield line.strip()


class TermClassifier:
    def __init__(self, term_list: Iterable):
        self.term_list = term_list
        self.term_frequency_dict = {}  # term: frequency
        self.term_name_space = set()  # kind of terms
        self.length = 0  # length of all terms

    def sort(self):
        for term in self.term_list:
            self.term_frequency_dict.setdefault(term, 0)
            self.term_frequency_dict[term] += 1
            self.term_name_space.add(term)
            self.length += 1



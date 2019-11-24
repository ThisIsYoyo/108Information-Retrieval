from pathlib import Path
from typing import Union, List

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class BaseLoader:
    def __init__(self, path: Path = None, file_name: str = None, file_path: Path = None):
        if file_path:
            self.real_file_path = file_path
        else:
            self.real_file_path = path / file_name
        self.start_line = 0


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


class BGLMLoader(BaseLoader):
    def __init__(self, path: Path, file_name: str):
        super(BGLMLoader, self).__init__(path, file_name)
        self._bglm_list = []

    def parse(self):
        with open(str(self.real_file_path), 'r') as fp:
            line_list = fp.readlines()

        for line in line_list:
            index, bglm = line.split()
            self._bglm_list.append([float(bglm)])

    def get_bglm_list(self):
        return self._bglm_list


class FreqMatrixTransformer:

    def __init__(self):
        self.x_axis = []
        self.y_axis = []
        self._doc_content_start_line = 0
        self.freq_matrix = None

    def set_x_axis(self, x_list: List = None, x_flie: Path = None):
        if x_flie:
            x_axis_loader = ListLoader(file_path=x_flie)
            self.x_axis = x_axis_loader.get_list()
        else:
            self.x_axis = x_list

    def set_y_axis(self, y_list: List = None, y_flie: Path = None):
        if y_flie:
            y_axis_loader = ListLoader(file_path=y_flie)
            self.y_axis = y_axis_loader.get_list()
        else:
            self.y_axis = y_list

    def set_doc_content_start_line(self, start_line):
        self._doc_content_start_line = start_line

    def vectorize(self, folder: Path):
        assert self.x_axis != []
        x_len = len(self.x_axis)

        if self.y_axis:
            y_len = len(self.y_axis)

            self.freq_matrix = np.zeros((y_len, x_len))
            for x, x_doc in enumerate(self.x_axis):
                x_term_loader = TermLoader(path=folder, file_name=x_doc)
                x_term_loader.set_start_line(self._doc_content_start_line)

                for x_term in x_term_loader.iter_term():
                    if x_term not in self.y_axis:
                        continue

                    y = self.y_axis.index(x_term)
                    self.freq_matrix[y, x] += 1

        else:
            doc_content_list = []
            for x_doc in self.x_axis:
                x_term_loader = TermLoader(path=folder, file_name=x_doc)
                x_term_loader.set_start_line(self._doc_content_start_line)

                x_content = ' '.join(list(x_term_loader.iter_term()))
                doc_content_list.append(x_content)

            cv = CountVectorizer()
            sparse_doc_matrix = cv.fit_transform(doc_content_list)
            self.freq_matrix = np.transpose(sparse_doc_matrix.toarray())
            self.y_axis = cv.get_feature_names()

        return self.freq_matrix



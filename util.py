from pathlib import Path


class BaseLoader:
    def __init__(self, path: Path, file_name: str):
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


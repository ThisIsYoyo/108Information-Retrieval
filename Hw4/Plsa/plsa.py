from pathlib import Path

from util import FreqMatrixTransformer

DATA = Path(__file__).parent.parent / 'Data'

freq_matrix = FreqMatrixTransformer()
freq_matrix.set_x_axis(x_flie=DATA / 'doc_list.txt')
freq_matrix.set_doc_content_start_line(3)
freq_matrix.vectorize(folder=DATA / 'Document')


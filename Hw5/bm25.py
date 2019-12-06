from os import listdir
from pathlib import Path
from rank_bm25 import BM25Okapi

DATA = Path(__file__).parent / 'Data'
DOC = DATA / 'doc'
TEST = DATA / 'test'

# -------- produce doc tokens --------

docs = listdir(DOC)
all_doc_tokens = []
for doc in docs:
    with open(DOC / doc, 'r') as fp:
        raw_doc_lines = fp.readlines()
    doc_lines = [doc_line.strip() for doc_line in raw_doc_lines]

    single_doc_tokens = []
    for doc_line in doc_lines:
        single_doc_tokens.extend(doc_line.split(' '))

    all_doc_tokens.append(single_doc_tokens)

# -------- make bm25 --------

bm25 = BM25Okapi(all_doc_tokens)

# -------- load query --------

with open(TEST / 'query_list.txt') as fp:
    raw_querys = fp.readlines()
query_list = [query.strip() for query in raw_querys]

# -------- ranking --------

doc_len = len(docs)

submission = open('submission.csv', 'w')
submission.write('Query,RetrievedDocuments')

for query in query_list:
    submission.write(f'\n{query},')

    with open(TEST / 'query' / query, 'r') as fp:
        query_first_line = fp.readline()
    query_tokens = query_first_line.strip().split(' ')

    doc_scores = bm25.get_scores(query_tokens)

    doc_scores_with_name = [(doc_scores[i], docs[i]) for i in range(doc_len)]
    sorted_doc_scores_with_name = sorted(doc_scores_with_name, reverse=True)
    ranked_doc = [score_with_name[1] for score_with_name in sorted_doc_scores_with_name]

    for doc in ranked_doc[:100]:
        submission.write(f'{doc} ')









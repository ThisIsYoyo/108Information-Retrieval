import time
from os import listdir
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F
from transformers import *
from tqdm import tqdm


# fixed path
from util import LineLoader

DATA = Path(__file__).parent / 'DATA'
DOC = DATA / 'doc'
TRAIN = DATA / 'train'
TEST = DATA / 'test'

# Setting
LR = 0.001


class Net(torch.nn.Module):

    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function for hidden layer

        x = self.out(x)
        return x


class SimpleBertIR(torch.nn.Module):

    def __init__(self):
        super(SimpleBertIR, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.Out_FC = torch.nn.Linear(768, 1)

    def forward(self, _input_ids):
        outputs = self.bert(input_ids=_input_ids.squeeze(1))
        CLS_representation = outputs[0][0][0]
        Pred_out = self.Out_FC(CLS_representation)

        return Pred_out


def read_query_and_doc(query_folder, query_name, doc_name):

    with open(query_folder / 'query' / query_name, 'r') as fp:
        query_lines = fp.readlines()
    query_content = query_lines[0].strip()

    doc_loader = LineLoader(path=DOC, file_name=doc_name)
    doc_content = doc_loader.read_all_as_line()

    return query_content, doc_content


if __name__ == '__main__':
    # net_SGD = Net(n_feature=2, n_hidden=10, n_output=2)
    # print(net_SGD)
    #
    # opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    #
    # loss_func = torch.nn.CrossEntropyLoss()
    #
    # # ----------------------------------------------------
    #
    # for epoch in tqdm(range(100)):
    #     out = net_SGD(x)  # x not set
    #     loss = loss_func(out, y)  # y not set
    #
    #     opt_SGD.zero_grad()
    #     loss.backward()
    #     opt_SGD.step()
    #
    # ----------------------------------------------------

    device = torch.device('cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # ----------------------------------------------------

    MSEloss = torch.nn.MSELoss()
    model = SimpleBertIR().to(device)
    model.train()
    PosAns = torch.tensor([1.]).to(device)
    NegAns = torch.tensor([0.]).to(device)

    optimizerSGD = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    with open(TRAIN / 'Pos.txt', 'r') as fp:
        pos_lines = fp.readlines()

    with open(TRAIN / 'Neg.txt', 'r') as fp:
        neg_lines = fp.readlines()

    # ---------------pos---------------

    pos_len = len(pos_lines)

    print('-----Pos part start-----')

    pos_start = time.time()
    for pos_line in pos_lines:
        query_name, doc_name, relav = pos_line.strip().split()

        query_content, doc_content = read_query_and_doc(query_folder=TRAIN, query_name=query_name, doc_name=doc_name)
        doc_len = len(doc_content)

        # add tokenizer 512 for each doc len
        i = 0
        while i < doc_len:
            start = i
            i = i + 512 if i + 512 <= doc_len else doc_len

            while doc_content[i - 1] != ' ' and i != doc_len:
                i -= 1

            x_input_ids = torch.tensor([tokenizer.encode(
                f'[CLS]{query_content}[SEP]{doc_content[start:i]}[SEP]', add_special_tokens=True)]).to(device)

            model_out = model(x_input_ids)
            output = MSEloss(model_out, PosAns)
            output.backward()
            optimizerSGD.step()
            optimizerSGD.zero_grad()

        print(f'finish {(pos_lines.index(pos_line) + 1) / pos_len * 100} %')

    pos_end = time.time()
    print(f'pos part finish with {(pos_end - pos_start) // 60} mins {(pos_end - pos_start) % 60} secs')

    # ---------------neg---------------

    neg_len = len(neg_lines)

    print('-----Neg part start-----')

    neg_start = time.time()
    for neg_line in neg_lines:
        query_name, doc_name, relav = neg_line.strip().split()
        query_content, doc_content = read_query_and_doc(query_folder=TRAIN, query_name=query_name, doc_name=doc_name)
        doc_len = len(doc_content)

        # add tokenizer 512 for each doc len
        i = 0
        while i < doc_len:
            start = i
            i = i + 512 if i + 512 <= doc_len else doc_len

            while doc_content[i - 1] != ' ' and i != doc_len:
                i -= 1

            x_input_ids = torch.tensor([tokenizer.encode(
                f'[CLS]{query_content}[SEP]{doc_content[start:i]}[SEP]', add_special_tokens=True)]).to(device)

            model_out = model(x_input_ids)
            output = MSEloss(model_out, NegAns)
            output.backward()
            optimizerSGD.step()
            optimizerSGD.zero_grad()

        print(f'finish {(neg_lines.index(neg_line) + 1) / neg_len * 100} %')

    neg_end = time.time()
    print(f'neg part finish with {(neg_end - neg_start) // 60} mins {(neg_end - neg_start) % 60} secs')

    torch.save(model, 'model_01.pt')

    # ---------------ranking---------------

    with open(TEST / 'query_list.txt') as fp:
        raw_querys = fp.readlines()
    query_list = [query.strip() for query in raw_querys]

    docs = listdir(DOC)

    submission = open('submission.csv', 'w')
    submission.write('Query,RetrievedDocuments')
    for query in query_list:
        submission.write(f'\n{query},')
        doc_scores_with_name = []

        for doc in docs:
            query_content, doc_content = read_query_and_doc(query_folder=TEST, query_name=query, doc_name=doc)
            doc_len = len(doc_content)

            # add tokenizer 512 for each doc len
            result_sum = 0
            i = 0
            while i < doc_len:
                start = i
                i = i + 512 if i + 512 <= doc_len else doc_len

                while doc_content[i - 1] != ' ' and i != doc_len:
                    i -= 1

                x_input_ids = torch.tensor([tokenizer.encode(
                    f'[CLS]{query_content}[SEP]{doc_content[start:i]}[SEP]', add_special_tokens=True)]).to(device)

                result_sum += model.forward(x_input_ids)

            doc_scores_with_name.append((result_sum, doc))

        ranked_doc = [score_with_name[1] for score_with_name in sorted(doc_scores_with_name, reverse=True)]

        for doc in ranked_doc[:100]:
            submission.write(f'{doc} ')


import torch
import numpy as np
import torch.nn.functional as F
from transformers import *
from tqdm import tqdm

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
    MSEloss = torch.nn.MSELoss()

    # ----------------------------------------------------

    x_input_ids = torch.tensor([tokenizer.encode('[CLS] {query} [SEP] {doc} [SEP]', )])

    PosAns = torch.tensor([1.]).to(device)
    NegAns = torch.tensor([0.]).to(device)

    ans = PosAns

    model = SimpleBertIR().to(device)
    model.train()

    optimizerSGD = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    model_out = model(x_input_ids)
    output = MSEloss(model_out, ans)
    output.backward()
    optimizerSGD.step()
    optimizerSGD.zero_grad()

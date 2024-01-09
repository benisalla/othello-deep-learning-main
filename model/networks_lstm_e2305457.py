import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMs(nn.Module):
    def __init__(self, conf):
        super(LSTMs, self).__init__()
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_LSTM/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_input_seq = conf["len_input_seq"]
        self.hidden_dim = conf["LSTM_conf"]["hidden_dim"]

        self.lstm = nn.LSTM(self.board_size * self.board_size, self.hidden_dim, batch_first=True)

        self.hidden2output = nn.Linear(self.hidden_dim * 2, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, seq):
        seq = torch.flatten(seq, start_dim=2)
        lstm_out, (hn, cn) = self.lstm(seq)
        out = self.hidden2output(torch.cat((hn, cn), -1))
        out = F.relu(out)
        return F.softmax(out, dim=1).squeeze() # PROBABILITY DIST

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

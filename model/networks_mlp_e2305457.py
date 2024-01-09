import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, conf):
        super(MLP, self).__init__()
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_MLP/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_input_seq = conf["len_input_seq"]

        self.lin1 = nn.Linear(self.board_size * self.board_size, 350)
        self.lin2 = nn.Linear(350, 350)
        self.lin3 = nn.Linear(350, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, seq):
        seq = torch.flatten(seq, start_dim=1)
        x = self.lin1(seq)
        x = self.lin2(F.relu(x))
        out = self.lin3(F.relu(x))
        return F.softmax(out, dim=-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

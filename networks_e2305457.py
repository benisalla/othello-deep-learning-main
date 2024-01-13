import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, conf):
        super(CNN, self).__init__()
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_CNN"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_input_seq"]

        # Communication: features extraction
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2)

        # Computation: learning the logic
        self.fc1 = nn.Linear(256 * 1 * 1, 128)
        self.fc2 = nn.Linear(128, self.board_size * self.board_size)

    def forward(self, seq):
        x = F.relu(self.conv1(seq))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        x = self.fc1(x.view(-1, 256 * 1 * 1))
        out = self.fc2(F.relu(x))
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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
        return F.softmax(out, dim=1).squeeze()  # PROBABILITY DIST

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

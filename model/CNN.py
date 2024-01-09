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

import copy
import os
import time
import h5py
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from Const import MOVE_DIRS


def test_model(chpts_path, data_path, data_dir, model_type, len_samples, device, batch_size):
    # load the test dataset
    dataset = CustomDataset(
        split_path=data_path,
        data_dir=data_dir,
        len_samples=len_samples)
    # create the data loader
    data_loader = DataLoader(dataset, batch_size=batch_size)

    # load the model
    model_path = "./pt_models/model_" + model_type + "/" + chpts_path
    model = torch.load(model_path)
    model.eval()
    acc = evaluate_model(model, data_loader, device)
    print(f"Accuracy[{model_type}]: {round(100 * acc, 2)}%")


def train_function(model, train_loader, dev_loader, num_epoch, device, optimizer):
    # Create directory if it doesn't exist
    if not os.path.exists(model.path_save):
        os.mkdir(model.path_save)

    best_dev = 0.0
    not_change = 0
    train_acc_list = []
    dev_acc_list = []
    torch.autograd.set_detect_anomaly(True)
    init_time = time.time()

    for epoch in range(1, num_epoch + 1):
        start_time = time.time()
        loss_batch = train_epoch(model, train_loader, device, optimizer)
        print(f"\nEpoch[{epoch}/{num_epoch}]: Loss: {loss_batch}\n")

        # Evaluate on training set
        acc_train = evaluate_model(model, train_loader, device)
        train_acc_list.append(acc_train)

        # Evaluate on development set
        acc_dev = evaluate_model(model, dev_loader, device)
        dev_acc_list.append(acc_dev)

        last_prediction = time.time() - start_time

        print(f"Accuracy Train: {round(100 * acc_train, 2)}%, Dev: {round(100 * acc_dev, 2)}%;",
              f"Time: {round(time.time() - init_time)}",
              f"(Last train: {round(last_prediction)}sec)")

        # Update best model
        if acc_dev > best_dev or best_dev == 0.0:
            not_change = 0
            save_model(model, epoch)
            best_dev = acc_dev
            best_epoch = epoch
        else:
            not_change += 1
            if not_change > model.earlyStopping:
                break

    print("\n", "*" * 15, f"\nThe best score on DEV {best_epoch}: {round(100 * best_dev, 3)}%")

    # Load the best model
    model = torch.load(model.path_save + f'/model_{best_epoch}.pt')

    # Evaluate on development set after training
    evaluate_model(model, dev_loader, device)


def train_epoch(model, train_loader, device, optimizer):
    model.train()
    loss_batch = 0
    nb_batch = 0

    for batch, labels, _ in tqdm(train_loader):
        outputs = model(batch.float().to(device))
        loss = loss_fun(outputs, labels.clone().detach().float().to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        nb_batch += 1
        loss_batch += loss.item()

    return loss_batch / nb_batch


def evaluate_model(model, data_loader, device):
    # Set the module in eval mode
    model.eval()
    all_predicts = []
    all_targets = []

    for data, target, _ in tqdm(data_loader, desc="Evaluating"):
        output = model(data.float().to(device))
        predicted = output.argmax(dim=-1).cpu().detach().numpy()
        target = target.argmax(dim=-1).numpy()

        all_predicts.extend(predicted)
        all_targets.extend(target)

    clas_rep = classification_report(all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)
    acc = clas_rep["weighted avg"]["recall"]

    return acc


def save_model(model, epoch):
    torch.save(model, model.path_save + f'/model_{epoch}.pt')


def isBlackWinner(move_array, board_stat, player=-1):
    move = np.where(move_array == 1)
    move = [move[0][0], move[1][0]]
    board_stat[move[0], move[1]] = player

    for direction in MOVE_DIRS:
        if has_tile_to_flip(move, direction, board_stat, player):
            i = 1
            while True:
                row = move[0] + direction[0] * i
                col = move[1] + direction[1] * i
                if board_stat[row][col] == board_stat[move[0], move[1]]:
                    break
                else:
                    board_stat[row][col] = board_stat[move[0], move[1]]
                    i += 1
    is_black_winner = sum(sum(board_stat)) < 0

    return is_black_winner


class CustomDataset(Dataset):
    def __init__(self, split_path, data_dir, len_samples):
        # init the board
        self.starting_board_stat = np.zeros((8, 8))
        self.starting_board_stat[3, 3] = -1
        self.starting_board_stat[4, 4] = -1
        self.starting_board_stat[3, 4] = +1
        self.starting_board_stat[4, 3] = +1

        self.split_path = split_path
        self.len_samples = len_samples
        self.path_dataset = data_dir

        # reading train/dev/test.txt files
        with open(self.split_path) as f:
            list_files = [line.rstrip() for line in f]
        self.game_files_name = list_files

        self.samples = np.zeros((len(self.game_files_name) * 30, self.len_samples, 8, 8), dtype=int)
        self.outputs = np.zeros((len(self.game_files_name) * 30, 8 * 8), dtype=int)
        idx = 0
        for gm_idx, gm_name in tqdm(enumerate(self.game_files_name)):
            h5f = h5py.File(self.path_dataset + gm_name, 'r')
            game_log = np.array(h5f[gm_name.replace(".h5", "")][:])
            h5f.close()
            last_board_state = copy.copy(game_log[0][-1])
            is_black_winner = isBlackWinner(game_log[1][-1], last_board_state)
            for sm_idx in range(30):
                if is_black_winner:
                    end_move = 2 * sm_idx
                else:
                    end_move = 2 * sm_idx + 1

                if end_move + 1 >= self.len_samples:
                    features = game_log[0][end_move - self.len_samples + 1:
                                           end_move + 1]
                else:
                    features = [self.starting_board_stat]
                    # Padding starting board state before first index of sequence
                    for i in range(self.len_samples - end_move - 2):
                        features.append(self.starting_board_stat)
                    # adding the initial of game as the end of sequence sample
                    for i in range(end_move + 1):
                        features.append(game_log[0][i])

                # if black is the current player the board should be multiplied by -1
                if is_black_winner:
                    features = np.array([features], dtype=int) * -1
                else:
                    features = np.array([features], dtype=int)

                self.samples[idx] = features
                self.outputs[idx] = np.array(game_log[1][end_move]).flatten()
                idx += 1

        print(f"Number of samples : {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features = self.samples[idx]
        y = self.outputs[idx]
        return features, y, self.len_samples


def loss_fun(predictions, targets):
    return nn.CrossEntropyLoss()(input=predictions, target=targets)


def is_legal_move(move, board_stat, NgBlackPsWhite):
    if move != () and is_valid_coord(move[0], move[1]) \
            and board_stat[move[0]][move[1]] == 0:
        for direction in MOVE_DIRS:
            if has_tile_to_flip(move, direction, board_stat, NgBlackPsWhite):
                return True
    return False


def get_legal_moves(board_stat, NgBlackPsWhite):
    moves = []
    for row in range(len(board_stat)):
        for col in range(len(board_stat)):
            move = (row, col)
            if is_legal_move(move, board_stat, NgBlackPsWhite):
                moves.append(move)
    return moves


def is_valid_coord(row, col, board_size=8):
    if 0 <= row < board_size and 0 <= col < board_size:
        return True
    return False


def has_tile_to_flip(move, direction, board_stat, NgBlackPsWith):
    i = 1
    if is_valid_coord(move[0], move[1]):
        while True:
            row = move[0] + direction[0] * i
            col = move[1] + direction[1] * i
            if not is_valid_coord(row, col) or \
                    board_stat[row][col] == 0:
                return False
            elif board_stat[row][col] == NgBlackPsWith:
                break
            else:
                i += 1
    return i > 1

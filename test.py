import torch
from utile import test_model

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

batch_size = 1000
data_dir = "./dataset/"
chpts_path = "model_1.pt"
data_path = "./ds-splits/train.txt"

# testing CNN model
test_model(
    chpts_path=chpts_path,
    data_path=data_path,
    data_dir=data_dir,
    model_type="CNN",
    len_samples=1,
    device=device,
    batch_size=batch_size)

# testing LSTM model
test_model(
    chpts_path=chpts_path,
    data_path=data_path,
    data_dir=data_dir,
    model_type="LSTM",
    len_samples=10,
    device=device,
    batch_size=batch_size)

# testing MLP model
test_model(
    chpts_path=chpts_path,
    data_path=data_path,
    data_dir=data_dir,
    model_type="MLP",
    len_samples=1,
    device=device,
    batch_size=batch_size)

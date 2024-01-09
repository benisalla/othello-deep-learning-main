import torch
from torch.utils.data import DataLoader
from Const import BOARD_SIZE
from model.MLP import MLP
from utile import CustomDataset, train_function

##############################################################
#          choosing on which device to run (GPU or CPU)      #
##############################################################
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print('Running on ' + str(device))

##############################################################
#                   hyperparameters                          #
##############################################################
lr = 0.001
num_epoch = 100
early_stop = 20
batch_size = 1000
len_samples = 1

##############################################################
#                           data params                      #
##############################################################
data_dir = "../dataset/"
chpts_path = "../pt_models/model"
dev_path = "../ds-splits/dev.txt"
train_path = "../ds-splits/train.txt"

##############################################################
#                       create datasets                      #
##############################################################
train_ds = CustomDataset(split_path=train_path,
                         data_dir=data_dir,
                         len_samples=len_samples)

dev_ds = CustomDataset(split_path=dev_path,
                       data_dir=data_dir,
                       len_samples=len_samples)

##############################################################
#                   create dataset loader                    #
##############################################################
train_loader = DataLoader(train_ds, batch_size=batch_size)
dev_loader = DataLoader(dev_ds, batch_size=batch_size)

##############################################################
#             Now it is time to initialize our model         #
##############################################################
Config = {
    "board_size": BOARD_SIZE,
    "path_save": chpts_path,
    'num_epoch': num_epoch,
    "earlyStopping": early_stop,
    "len_input_seq": len_samples,
}

model = MLP(Config).to(device)

# let's see number of params in our little model
print("\nNumber of parameters: %s\n" % model.count_parameters())

##############################################################
#                      training configuration                #
##############################################################
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr,
)

# run training
train_function(model, train_loader, dev_loader, num_epoch, device, optimizer)

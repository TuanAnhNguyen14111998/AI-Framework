from model.model import Net
from model.loss import get_loss
from model.optimizers import get_optimizer
from model.data_loader import Dataset
from helpers.save_checkpoint.save_ckp import save_ckp
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import argparse
import json
import torch
from torch.utils import data
from tqdm import tqdm
import numpy as np
import csv
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to fordel save file information dataset (csv)")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to fordel save file class info (csv)")
ap.add_argument("-i_w", "--image_width", type=int, default=32,
	help="image size")
ap.add_argument("-i_h", "--image_height", type=int, default=32,
	help="image size")
ap.add_argument("-ep", "--epoch", type=int, default=50,
	help="number of epochs")
ap.add_argument("-bz", "--batch_size", type=int, default=32,
	help="number of epochs")
ap.add_argument("-m_acc", "--max_accuracy", type=float, default=1000,
	help="min val loss")
ap.add_argument("-n_class", "--number_class", type=int, default=13,
	help="number of class")
ap.add_argument("-t", "--test", type=str, default="false",
	help="option use test set")
args = vars(ap.parse_args())

input_fordel = args["input"]
output_fordel = args["output"]
use_test = args["test"]

print("Starting train model classification ... ")

df_train = pd.read_csv(input_fordel + "/information_training.csv")
df_val = pd.read_csv(input_fordel + "/information_validation.csv")
if use_test == "true":
    df_test = pd.read_csv(input_fordel + "/information_testing.csv")

f = open(input_fordel + "/information_class.json") 
labels = json.load(f)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Parameters
image_size = (args["image_width"], args["image_height"])
max_epochs = args["epoch"]

params = {
    'batch_size': args["batch_size"],
    'shuffle': True,
    'num_workers': 6
}

# Generators
training_set = Dataset(df_train, labels, image_size)
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(df_val, labels, image_size)
validation_generator = data.DataLoader(validation_set, **params)

net = Net(n_class=args["number_class"], input_size=image_size[0])
net.cuda()
optimizer = get_optimizer(net, lr=0.001, momentum=0.9)

valid_acc_max = args["max_accuracy"]

dir_name = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = dir_name + "/weights/current_checkpoint.pt"
best_model_path = dir_name + "/weights/best_model.pt"

history_file = open(dir_name + "/weights/history.txt", "a")
history_file.write("epoch,val_loss,val_acc\n")
history_file.close()

# Loop over epochs
for epoch in range(max_epochs):
    # Training
    number_iter = 1
    correct = 0
    total = 0
    train_loss = []

    ###################
    # train the model #
    ###################
    net.train()
    with tqdm(training_generator, unit="batch") as tepoch:
        for local_batch, local_labels in tepoch:
            tepoch.set_description("Epoch {}, interation {}".format(epoch, number_iter))
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(local_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += local_labels.size(0)
            correct += (predicted == local_labels).sum().item()
            loss = get_loss(outputs, local_labels)
            loss.backward()
            optimizer.step()

            train_loss.extend([loss.item()])

            tepoch.set_postfix(loss=loss.item())
            number_iter += 1
        train_loss = np.mean(np.array(train_loss))

    train_acc = (100 * correct / total)

    ######################    
    # validate the model #
    ######################
    correct = 0
    total = 0
    val_losses = []
    with torch.no_grad():
        for local_batch, local_labels in tqdm(validation_generator):
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            # forward + backward + optimize
            outputs = net(local_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += local_labels.size(0)
            correct += (predicted == local_labels).sum().item()
            loss = get_loss(outputs, local_labels)

            val_losses.extend([loss.item()])
        val_losses = np.mean(np.array(val_losses))
    
    val_acc = (100 * correct / total)
    
    print("Epoch {}, Train Loss {:.6f}, Validate loss {:.6f}, Train Accuracy {:.2f}, Validate Accuracy {:.2f}".format(epoch, train_loss, val_losses, train_acc, val_acc))

    checkpoint = {
        'epoch': epoch + 1,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    # save checkpoint
    save_ckp(checkpoint, False, checkpoint_path, best_model_path)

    ## TODO: save the model if validation loss has decreased
    if val_acc >= valid_acc_max:
        print("Validate accuracy decreased ({:.6f} --> {:.6f}).  Saving model ...".format(valid_acc_max, val_acc))
        # save checkpoint as best model
        save_ckp(checkpoint, True, checkpoint_path, best_model_path)
        valid_acc_max = val_acc
    
    history_file = open(dir_name + "/weights/history.txt", "a")
    history_file.write(",".join((str(epoch), str(val_losses), str(val_acc))) + "\n")
    history_file.close()

print("Done!")

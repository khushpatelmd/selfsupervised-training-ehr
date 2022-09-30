#@Author: Khush Patel, drpatelkhush@gmail.com
#Config options: change data/logging/checkpoint paths, experiment name, gpu, training options

#####################################  MOST LIKELY TO BE CHANGED  ##################################################
#device
gpu_number = ":" + "0"
device = torch.device('cuda' + gpu_number) if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')

#experiment name
experiment_name  = "bert01"

#Batch size
batch_size = 10

#####################################  Imports  ##################################################

#python imports
from tqdm import tqdm
import random
import os
import pickle
import numpy as np
from prettytable import PrettyTable
import sys
import dill
import logging
from sklearn.metrics import accuracy_score
import numpy as np
from statistics import mean
import shutil

#pytorch imports
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn

#transformers imports
import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

###################################     Paths     ##########################################################
#Directory
path_on_server="/home/Khush/" + experiment_name
if os.path.isdir(path_on_server):
    pass
else:
    os.mkdir(path_on_server)


#Data Location
raw_data_path = "../" + ".train"

#Saving model checkpoints
#Location
save_directory = path_on_server  + "/modelcheckpoints_"

if os.path.isdir(save_directory):
    pass
else:
    os.mkdir(save_directory)

save_dir = save_directory  + experiment_name + ".pt"
    
save_every_step = 5000

#Save metrics like loss and accuracy after how many steps
measure_metrics_steps = 100


#Saving model weights if performance improved then last epoch
#Location
save_wts_directory = path_on_server  + "/saved_weights_/"
save_wts_loc = save_wts_directory + "pytorch_model" + ".bin"
if os.path.isdir(save_wts_directory):
    pass
else:
    os.mkdir(save_wts_directory)

#tensorboard directory
tbpath = 'runs/' + experiment_name + "/"


#Saving model weigths if performance improved over the epoch

#logging
log_path = path_on_server +  "/modelcheckpoints_/" + "logging/"
if os.path.isdir(log_path):
    pass
else:
    os.mkdir(log_path)

logging_path = log_path+experiment_name+".log"

print(logging_path)

logging.basicConfig(filename=logging_path, level=logging.DEBUG, format="	%(asctime)s:%(message)s")

##############################      Hyperparameters      ######################################
#Learning Rate
lr = 5e-5
#Number of epochs
num_of_epochs = 20


#max_position_embeddings – The maximum sequence length that this model might ever be used with. 
max_position_embeddings = 128
#Encoding for mask tokens
masked_token_encoding = 103
#Percentage of tokens to be masked
percent_tokens_to_mask = 0.15
#Vocabulary size
vocab_size = 84840
#Number of patients
dataset_size = 2000000

########################Calculated parameters#######################################

#Number of training steps
num_train_steps = int(dataset_size / batch_size * num_of_epochs)


##################################    Seeds    ######################################################

#seeds: Change seeds for changing mask for MLM. Currently doing after 4 epochs manually

#seed for MLM masking
run_no_mask = 0  #Keep it fixed as 0 as I am using this to know the actual number of steps completed while resuming training.

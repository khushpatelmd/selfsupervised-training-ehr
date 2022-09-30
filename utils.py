from config import *

#Clear contents of logging file
a_file = open(logging_path, "w")
a_file.truncate()
a_file.close()
print("Cleared log file")
       
#Remove tb logging files
shutil.rmtree(tbpath)
print("Removed tb files")


#Get the stats for resuming training

checkpoint = torch.load(save_dir, pickle_module=dill)
epoch = checkpoint['epoch']
loss = checkpoint['loss']
number_of_training_steps = checkpoint['number_of_training_steps']
seed_value_changer = checkpoint["seed_value_changer"]

print(f"Number of epochs completed are {epoch}")
print(f"Number of training steps completed so far in that epoch are {number_of_training_steps}")
print(f"Number of epochs remaining are {num_of_epochs-epoch}")
print(f"Last recorded loss was {loss}")
print(f"The last seed used was {seed_value_changer}")

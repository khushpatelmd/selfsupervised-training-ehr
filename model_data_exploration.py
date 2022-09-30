from config import *
from model import *

# max sequence length size: 128: 4.47% data has length more than 128
# max sequence length size: 256: 1.11% data has length more than 256
# max sequence length size: 512: 0% data has length more than 512
# max length is 511

###512

# vocab length: 56589
# max vocab value: 84840


# Counting vocab size
with open(raw_data_path, "rb") as f:
    raw = pickle.load(f)
    
los = []
vocab = []
length = []
for i in raw:
    vocab.extend(i[3])
    length.append(len(i[3]))
    los.append(max(i[1]))
vocab = list(set(vocab))
print(f"The longest duration of stay was {max(los)} and was at position {los.index(max(los))}")

print(f"Number of unique code values present {len(vocab)}")  # 56589

# 84840
print(
    f"The max value of code to estimate vocab size for transformer config file {max(vocab)}")

print(f"Total number of training samples {len(raw)}")  # 20,00,000

# 125000
print(
    f"Total number of steps for one epoch {int(dataset_size / batch_size * 1)}")

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


count_parameters(model)

# Checking len of input ids


def length_of_input_ids(size=256, length=length, dataset_size=dataset_size):
    """Provide size of length of input ids and it will print percentage of data having size larger than 
    that size.
    For eg, size = 512
    Returned value will be 0% as there are no inputids for any patient of size greater than 512.

    Args:
        size ([int]): Length of input ids to get percentage of data having size larger than that size.
        length : the length list generated above
    """

    counter = 0
    for i in length:
        if i > size:
            counter += 1
    else:
        pass

    print(
        f"The percentage of data having size larger than embedding size of {size} is {(counter/dataset_size)*100} %")
    print(
        f"Also the maximum length of input ids for all patient was {max(length)}")


length_of_input_ids()

sys.exit()

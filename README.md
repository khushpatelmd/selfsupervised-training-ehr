# Self supervised pretraining of transformer based architecture for Electronic Health Records (EHR)

Millions of patients' electronic health records (EHR) are available through Optum, Cerner and other providers. Recently, transformer based architectures have showed tremendous potential in natural language processing tasks. Self supervised learning approaches such as masked language modeling (MLM) and next sentence prediction (NSP) have made possible to use millions of unlabelled text datasets. Similarly, a transformer model can be trained on millions of patients' unlabelled EHR data comprising of diagnoses codes (ICD-9 and ICD-10), procedures codes (CPT,HCPCS, and ICD PCS),and medications (Multum ID and multum categories). Each EHR code can be thought of as a word in sentence and each patient record as a sentence. The repository was built by me to enable any compatible tranformer based architecture pretraining on EHR data using masked language modeling and next sentence prediction. The original idea of this concept stems from medbert model from our lab where tensorflowv1 bert original code was used for training EHR data on diagnosis code.

Pretraining strategies:

Next sentence prediction task has been replaced by lenght of stay longer than 7 days (yes/no).

Masked language modeling (MLM): 
    A particular percentage of EHR codes (15% by default, value to be mentioned in config.py) for each patient (compared to a sentence) are masked randomly every    epoch. Out of the masked tokens, 10% of the EHR codes are replaced by random EHR codes, 10% of the codes are kept as original. 
    
    **Dynamic masking strategy:** 
    For Bert like models, every epoch, the codes to be masked are selected randomly.
    
    **Fixed masking strategy:** 
    For Roberta like models, the same EHR codes should be masked for 4 epochs, then changed every 4 epochs. A training strategy has been developed for the same. (Although no advantage was seen in the actual results)        
    




# Requirements
- [yacs](https://github.com/rbgirshick/yacs) (Yet Another Configuration System)
- [PyTorch](https://pytorch.org/) (An open source deep learning platform) 
- [ignite](https://github.com/pytorch/ignite) (High-level library to help with training neural networks in PyTorch)

# Table Of Contents
-  [In a Nutshell](#in-a-nutshell)
-  [In Details](#in-details)
-  [Future Work](#future-work)
-  [Contributing](#contributing)
-  [Acknowledgments](#acknowledgments)

# In a Nutshell   
In a nutshell here's how to use this template, so **for example** assume you want to implement ResNet-18 to train mnist, so you should do the following:
- In `modeling`  folder create a python file named whatever you like, here we named it `example_model.py` . In `modeling/__init__.py` file, you can build a function named `build_model` to call your model

```python
from .example_model import ResNet18

def build_model(cfg):
    model = ResNet18(cfg.MODEL.NUM_CLASSES)
    return model
``` 

   
- In `engine`  folder create a model trainer function and inference function. In trainer function, you need to write the logic of the training process, you can use some third-party library to decrease the repeated stuff.

```python
# trainer
def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn):
 """
 implement the logic of epoch:
 -loop on the number of iterations in the config and call the train step
 -add any summaries you want using the summary
 """
pass

# inference
def inference(cfg, model, val_loader):
"""
implement the logic of the train step
- run the tensorflow session
- return any metrics you need to summarize
 """
pass
```

- In `tools`  folder, you create the `train.py` .  In this file, you need to get the instances of the following objects "Model",  "DataLoader”, “Optimizer”, and config
```python
# create instance of the model you want
model = build_model(cfg)

# create your data generator
train_loader = make_data_loader(cfg, is_train=True)
val_loader = make_data_loader(cfg, is_train=False)

# create your model optimizer
optimizer = make_optimizer(cfg, model)
```

- Pass the all these objects to the function `do_train` , and start your training
```python
# here you train your model
do_train(cfg, model, train_loader, val_loader, optimizer, None, F.cross_entropy)
```

**You will find a template file and a simple example in the model and trainer folder that shows you how to try your first model simply.**


# In Details
```
├──  config
│    └── defaults.py  - here's the default config file.
│
│
├──  configs  
│    └── train_mnist_softmax.yml  - here's the specific config file for specific model or dataset.
│ 
│
├──  data  
│    └── datasets  - here's the datasets folder that is responsible for all data handling.
│    └── transforms  - here's the data preprocess folder that is responsible for all data augmentation.
│    └── build.py  		   - here's the file to make dataloader.
│    └── collate_batch.py   - here's the file that is responsible for merges a list of samples to form a mini-batch.
│
│
├──  engine
│   ├── trainer.py     - this file contains the train loops.
│   └── inference.py   - this file contains the inference process.
│
│
├── layers              - this folder contains any customed layers of your project.
│   └── conv_layer.py
│
│
├── modeling            - this folder contains any model of your project.
│   └── example_model.py
│
│
├── solver             - this folder contains optimizer of your project.
│   └── build.py
│   └── lr_scheduler.py
│   
│ 
├──  tools                - here's the train/test model of your project.
│    └── train_net.py  - here's an example of train model that is responsible for the whole pipeline.
│ 
│ 
└── utils
│    ├── logger.py
│    └── any_other_utils_you_need
│ 
│ 
└── tests					- this foler contains unit test of your project.
     ├── test_data_sampler.py
```


# Future Work

# Contributing
Any kind of enhancement or contribution is welcomed.


# Acknowledgments




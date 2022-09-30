# Self supervised pretraining of transformer based architecture for Electronic Health Records (EHR)

Millions of patients' electronic health records (EHR) coded data are available through Optum, Cerner and other providers. Recently, transformer based architectures have showed tremendous potential in natural language processing tasks. Self supervised learning approaches such as masked language modeling (MLM) and next sentence prediction (NSP) have made possible to use millions of unlabelled text datasets. Similarly, a transformer model can be trained on millions of patients' unlabelled EHR data comprising of diagnoses codes (ICD-9 and ICD-10), procedures codes (CPT,HCPCS, and ICD PCS),and medications (Multum ID and multum categories). Each EHR code can be thought of as a word in sentence and each patient record as a sentence. The repository was built by me to enable any compatible tranformer based architecture pretraining on EHR data using masked language modeling and next sentence prediction. The original idea of this concept stems from medbert model from our lab where tensorflowv1 bert original code was used for training EHR data on diagnosis code.

Please note data or model cannot be shared due to HIPAA compliance. For finetuning code or questions about pretraining code, contact Khush Patel at drpatelkhush@gmail.com 

<hr />

# Table Of Contents
-  [Pretraining strategies](## Pretraining strategies:)
-  [In Details](#in-details)
-  [Future Work](#future-work)
-  [Contributing](#contributing)
-  [Acknowledgments](#acknowledgments)


## Pretraining strategies:

### Next sentence prediction task (NSP) has been replaced by lenght of stay longer than 7 days (yes/no).

### Masked language modeling (MLM): 

A particular percentage of EHR codes (15% by default, value to be mentioned in config.py) for each patient (compared to a sentence) are masked randomly every    epoch. Out of the masked tokens, 10% of the EHR codes are replaced by random EHR codes, 10% of the codes are kept as original. 
    
#### Dynamic masking strategy:
For Bert like models, every epoch, the codes to be masked are selected randomly.
    
#### Fixed masking strategy:
For Roberta like models, the same EHR codes should be masked for 4 epochs, then changed every 4 epochs. A training strategy has been developed for the same. (Although no advantage was seen in the actual results)        
    
<hr />

### How to run

Step 1. Go to model.py and define the model you want (eg ). You need vocab size (default=90,000), maximum sequence length (all codes for a patient, default=256),  vocabulary size of the token_type_ids (default=1000), other model configs (num of attention heads, etc). Run model_data_exploration.py to count number of parameters as well as get information of the dataset. 

Step 2. Go to config.py to change data, logging and checkpoint paths, change experiment name, gpu number and batch size depending on the gpu you are working. MUltiple configurable options such as when to save checkpoints, learning rate, number of epochs, masking criteria, vocab size, dataset size, seed for fixed masking (eg Roberta). Please go through all options once.

Step 3. 
For fixed masking: 
To run MLM and NSP tasks for pretraining, run train_fixed_mask_nsp.py. To run only MLM task for pretraining, run train_fixed_mask.py. For dynamic masking:
To run dynamic masking MLM tasks for pretraining, run train_dynamic_mask.py

Step 4. 
To resume training: for MLM and NSP tasks for pretraining, run resume_train_fixed_mask_nsp.py. To resume running only MLM task for pretraining, run resume_train_fixed_mask.py. To resume training for dynamic masking MLM tasks for pretraining, run resume_train_dynamic_mask.py

Tensorboard is used for logging.




# Requirements
- [yacs](https://github.com/rbgirshick/yacs) (Yet Another Configuration System)
- [PyTorch](https://pytorch.org/) (An open source deep learning platform) 
- [ignite](https://github.com/pytorch/ignite) (High-level library to help with training neural networks in PyTorch)




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




# Self supervised pretraining of transformer based architecture for Electronic Health Records (EHR)

Millions of patients' electronic health records (EHR) coded data are available through Optum, Cerner and other providers. Recently, transformer based architectures have showed tremendous potential in natural language processing tasks. Self supervised learning approaches such as masked language modeling (MLM) and next sentence prediction (NSP) have made possible to use millions of unlabelled text datasets. Similarly, a transformer model can be trained on millions of patients' unlabelled EHR data comprising of diagnoses codes (ICD-9 and ICD-10), procedures codes (CPT,HCPCS, and ICD PCS),and medications (Multum ID and multum categories). Each EHR code can be thought of as a word in sentence and each patient record as a sentence. The repository was built by me to enable any compatible tranformer based architecture pretraining on EHR data using masked language modeling and next sentence prediction. The original idea of this concept stems from medbert model from our lab where tensorflowv1 bert original code was used for training EHR data on diagnosis code.

The pretrained model can then be finetuned for any clinical condition. We saw far superior results for predicting pancreatic cancer and heart failure patient outcomes vs training model from scratch for the condition.

Please note data or model cannot be shared due to HIPAA compliance. For finetuning code or questions about pretraining code, contact Khush Patel, MD at drpatelkhush@gmail.com 

<hr />

# Table Of Contents
-  [Pretraining strategies](#Pretraining-strategies)
-  [How to run the code](#How-to-run)
-  [Code structure](#Code-structure)
-  [Requirements](#Requirements)
-  [How to cite](#How-to-cite)

<hr />

# Pretraining strategies

### Next sentence prediction task (NSP) has been replaced by lenght of stay longer than 7 days (yes/no).

### Masked language modeling (MLM): 

A particular percentage of EHR codes (15% by default, value to be mentioned in config.py) for each patient (compared to a sentence) are masked randomly every    epoch. Out of the masked tokens, 10% of the EHR codes are replaced by random EHR codes, 10% of the codes are kept as original. 
    
#### Dynamic masking strategy:
For Roberta like models, every 4 epochs, the EHR codes to be masked are selected randomly. The number of epochs after which masking is changed is configurable.
    
#### Fixed / static masking strategy:
For original BERT model, the EHR codes are masked only once during data preprocessing which means the same input masks are fed to the model on epoch. A training strategy has been developed for the same.    

#### Data: The EHR data is a list of list where the structure is [[patient_id],[lenght of stay],[time between two visits],[EHR_codes(ICD/CPT,etc) mapped as per vocab file],[visit number]]
    
<hr />

# How to run

Step 1. Go to model.py and define the model you want (eg ). You need vocab size (default=90,000), maximum sequence length (all codes for a patient, default=256),  vocabulary size of the token_type_ids (default=1000), other model configs (num of attention heads, etc). Run model_data_exploration.py to count number of parameters as well as get information of the dataset. 

Step 2. Go to config.py to change data, logging and checkpoint paths, change experiment name, gpu number and batch size depending on the gpu you are working. MUltiple configurable options such as when to save checkpoints, learning rate, number of epochs, masking criteria, vocab size, dataset size, seed for fixed masking (eg Roberta). Please go through all options once.

Step 3. 
For fixed masking: 
To run MLM and NSP tasks for pretraining, run train_fixed_mask_nsp.py. To run only MLM task for pretraining, run train_fixed_mask.py. For dynamic masking:
To run dynamic masking MLM tasks for pretraining, run train_dynamic_mask.py

Step 4. 
To resume training: for MLM and NSP tasks for pretraining, run resume_train_fixed_mask_nsp.py. To resume running only MLM task for pretraining, run resume_train_fixed_mask.py. To resume training for dynamic masking MLM tasks for pretraining, run resume_train_dynamic_mask.py

Tensorboard and csv logging is used. Automatic Mixed Precision is used to allow pretraining on any sized GPU.

<hr />

# Code structure
```
├──  configs
│    └── config.py - change data/logging/checkpoint paths, experiment name, gpu, training options, hyperparameters
│
├──  data  
│    └── dataset_MLM_NSP.py - dataset class for MLM and NSP pretraining tasks
│    └── dataset_MLM.py - dataset class for MLM pretraining tasks
│
├──  engine - The training function to be used in the training files.
│   ├── engine_amp_MLM_NSP.py  - Main training loop for MLM and NSP task to be used inside train_*.py files
│   └── engine_amp_MLM.py  - Main training loop for MLM task
│
├── train - this folder contains main training files. 
│   └── train_dynamic_mask.py - Main training file for dynamic MLM strategy
│   └── train_fixed_mask_nsp.py - Main training file for fixed MLM strategy and NSP task
│   └── train_fixed_mask.py - Main training file for fixed MLM strategy 
│   └── resume_train_dynamic_mask.py - resuming training file for dynamic MLM strategy from last checkpoint
│   └── resume_train_fixed_mask_nsp.py - resuming training file for fixed MLM strategy and NSP task from last checkpoint
│   └── resume_train_fixed_mask.py - resuming training file for fixed MLM strategy from last checkpoint
│ 
├──  tools                - here's the train/test model of your project.
│    └── train_net.py  - here's an example of train model that is responsible for the whole pipeline.
│ 
└── utils
     └── utils.py - misc utils 
     └── requirements.txt - python libraries
     
```
<hr />

# Requirements
The `requirements.txt` file contains all Python libraries and they will be installed using:
```
pip install -r requirements.txt
```
    -torch==1.8.1
    -torch-tb-profiler==0.1.0
    -torchaudio==0.8.1
    -torchvision==0.9.1
    -prettytable==2.1.0
    -tensorboard==2.5.0
    -tensorboard-data-server==0.6.1
    -tensorboard-plugin-wit==1.8.0
    -transformers==4.6.1
    -tqdm=4.59.0=pyhd3eb1b0_1
    -numpy=1.20.1=py38h34a8a5c_0
    -numpy-base=1.20.1=py38haf7ebc8_0
    -numpydoc=1.1.0=pyhd3eb1b0_1

<hr />

# How to cite
This repository is a research work in progress. Please contact author (drpatelkhush@gmail.com) for details on reuse of code.



import os
import sys
import math
from tqdm import tqdm
from datetime import datetime
import ipdb
from typing import List, Dict, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

# huggingface
import transformers
from datasets import load_dataset, load_from_disk

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.cuda.empty_cache()

# hyperparameters
batch_size = 1
epochs = 3
lr = 6e-5
lr_warmup_steps = 100
contex = 1024
alpha = 0.5 #scaling factor for the ORPO odds ratio
prompt_max_size = 512 # limit for the prompt part of the interaction
compile_ = False
dtype = torch.bfloat16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
log_inter = 50

dropout = 0.
grad_clip = 1.0
weight_decay = 0.0

# logging
project_name = "alignment"
wandb_log = False
wandb_project = "alignment"
wandb_run_name = "alignment" + datetime.now().strftime("%Y%m%d%H%M%S")

if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name)
    print("wandb logging enabled")

dataset_path = os.path.join("data", "orpo_dataset")
dataset_name = "mlabonne/orpo-dpo-mix-40k"
tokenizer_path = os.path.join("tokenizers", "tok16384")
checkpoint_dir = "models"

# load the tokenizer in hugingface format
tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

# set our interaction template
tokenizer.chat_template



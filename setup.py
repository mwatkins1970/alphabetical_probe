GD_PATH = '/content/Drive'

from google.colab import drive
drive.mount(GD_PATH)

!pip install transformers
!pip install accelerate
!pip install torch torchvision

import os
import random
import json
import pickle	

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import GPT2Tokenizer, GPT2LMHeadModel, utils, AutoTokenizer, GPTJForCausalLM

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
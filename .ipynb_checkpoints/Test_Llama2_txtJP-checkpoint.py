import numpy as np
import torch
import random
import csv

from transformers import AutoModelForCausalLM, AutoTokenizer
import sys


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


device_0 = torch.device('cuda:7')   # device_1 for LLaMA Guard 2

random_number = random.randint(1, 2000)
random.seed(random_number)
# Set the random seed for NumPy
np.random.seed(random_number)
# Set the random seed for PyTorch
torch.manual_seed(random_number)
# If you are using CUDA (i.e., a GPU), also set the seed for it
torch.cuda.manual_seed_all(random_number)

#=============================================================Load LLaMA2 Chat Model=========================================================================#
# model_chat_path = "../save_models/Llama_2_7b_chat_hf"
tokenizer_path = "../save_models/Llama_2_7b_chat_hf"
model_chat = AutoModelForCausalLM.from_pretrained(
            "../save_models/Llama_2_7b_chat_hf",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=False
        ).to(device_0).eval()
    
tokenizer_chat = AutoTokenizer.from_pretrained(
    "../save_models/Llama_2_7b_chat_hf",
    trust_remote_code=True,
    use_fast=False
)

if 'oasst-sft-6-llama-30b' in tokenizer_path:
    tokenizer_chat.bos_token_id = 1
    tokenizer_chat.unk_token_id = 0
if 'guanaco' in tokenizer_path:
    tokenizer_chat.eos_token_id = 2
    tokenizer_chat.unk_token_id = 0
if 'Llama' in tokenizer_path:
    tokenizer_chat.pad_token = tokenizer_chat.unk_token
    tokenizer_chat.padding_side = 'left'
if 'falcon' in tokenizer_path:
    tokenizer_chat.padding_side = 'left'
if not tokenizer_chat.pad_token:
    tokenizer_chat.pad_token = tokenizer_chat.eos_token

text = [1, 14350, 263, 12618, 1400, 393, 18443, 267, 13736, 21448, 322, 20676, 9257, 13802, 10511, 5352, 29649, 4618, 30895, 15620]
text = torch.tensor(text)
print(text)
completion_top1 = \
tokenizer_chat.batch_decode(text, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(completion_top1)
exit()
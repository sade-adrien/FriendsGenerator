''' Parameters and global variables '''
import torch
import os

base_directory = os.getcwd()
file_path = os.path.join(base_directory, 'friends_script.txt')

batch_size = 128
block_size = 300 #max context size
max_iters = 20000 #training loop steps
eval_interval = 500 #every steps we eval loss
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 #number of random batches used to eval loss
n_embd = 128 #number of embedding dimension
n_heads = 4 #number of attention heads
n_blocks = 6 #number of unit blocks in transformer
dropout = .1 #dropout in dropout layers in training

with open(file_path, 'r') as file:
  text = file.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

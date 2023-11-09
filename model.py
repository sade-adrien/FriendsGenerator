import torch
import torch.nn as nn
import torch.nn.functional as F
import parameters


class GPTLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(parameters.vocab_size, parameters.n_embd)   #embedding token
    self.position_embedding_table = nn.Embedding(parameters.block_size, parameters.n_embd) #create an embedding for position idx too
    self.transformer = nn.Sequential(*[Block(n_embd=parameters.n_embd, n_heads=parameters.n_heads) for _ in range(parameters.n_blocks)])
    self.layernorm_final = nn.LayerNorm(parameters.n_embd)
    self.language_modeling_head = nn.Linear(parameters.n_embd, parameters.vocab_size)     #linear layer

  def forward(self, idx, targets=None):
    #idx and targets are (B,T)
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx)  #(B,T,C) doesn't give directly logits as we go through intermediate dimension of embedding
    pos_emb = self.position_embedding_table(torch.arange(T, device=parameters.device)) #(T,C)
    x = tok_emb + pos_emb   #(B,T,C) pos_emb are broadcasted along B dim
    x = self.transformer(x) #(B,T,C)
    x = self.layernorm_final(x) #(B,T,C)
    logits = self.language_modeling_head(x)    #(B,T,vocab_size)

    if targets != None:
      B, T, C = logits.shape
      logits = logits.view(B*T,C) #(B*T,C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
      return logits, loss
    return logits, None
  
  def generate(self, idx, max_new_tokens):      #Used to create new text (not in training phase)
    #idx is (B,T)
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -parameters.block_size:]  #have to crop to the last block size of given context
      logits, loss = self(idx_cond) #get predictions
      logits = logits[:, -1, :] #focus on last time step (B,C) (history not used yet)
      probs = F.softmax(logits, dim=-1) #get probas (B,C)
      idx_next = torch.multinomial(probs, num_samples=1) #sample from the probs distribution (B,1)
      idx = torch.cat((idx, idx_next), dim=1) #append new index to running sequence (B,T+1)
    return idx
    
    
class Head(nn.Module):
  #One head of attention mechanism
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(parameters.n_embd, head_size, bias=False)
    self.query = nn.Linear(parameters.n_embd, head_size, bias=False)
    self.value = nn.Linear(parameters.n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(parameters.block_size, parameters.block_size))) #create the 'tril' variable from buffer as not a parameter of the model (lower triangular matrix of ones - will be useful for masking)
    self.dropout = nn.Dropout(parameters.dropout)

  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x) #(B,T,C)
    q = self.query(x) #(B,T,C)
    v = self.value(x) #(B,T,C)

    #compute the weights to be used in attention 
    weights = q @ k.transpose(-2,-1) / torch.sqrt(torch.tensor(C)).item() #(B,T,C) @ (B,C,T) = (B,T,T) (also the paper introduces a 1/sqrt(C) factor to avoid (one-hot)-like weights after softmax)
    weights = weights.masked_fill(self.tril[:T,:T] == 0, float('-inf')) #masking upper triangular values as we can't communicate with future (B,T,T)
    weights = F.softmax(weights, dim=-1) #softmax to normalize (as probas) (B,T,T)
    weights = self.dropout(weights)

    #finnaly aggregate the values using these weights
    out = weights @ v #(B,T,T) @ (B,T,C) = (B,T,C)
    return out
    

class MultiHead(nn.Module):
#Multiple head model
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.projection = nn.Linear(parameters.n_embd, parameters.n_embd)   #need to project the list back into shape for skip connection (+ with original x)
    self.dropout = nn.Dropout(parameters.dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.projection(out)
    #out = self.dropout(x)
    return out
    
    
class FeedForward(nn.Module):
#Final part of decoder-only-transformer block
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd, 4*n_embd),    #factor 4 comes from paper
        nn.ReLU(),
        nn.Linear(4*n_embd, n_embd),  #here to project the list back into shape for skip connection (+ with original x)
        nn.Dropout(parameters.dropout),
    )

  def forward(self, x):
    return self.net(x)
    

class Block(nn.Module):
#(decoder only)-Transformer unit block
  def __init__(self, n_embd, n_heads):
    super().__init__()
    head_size = n_embd // n_heads
    self.multi_attention = MultiHead(n_heads, head_size)
    self.feedforward = FeedForward(n_embd)
    self.layernorm1 = nn.LayerNorm(n_embd)
    self.layernorm2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.multi_attention(self.layernorm1(x))  #with skip connection (layer normalization is applied before skip connection unlike in the paper)
    x = x + self.feedforward(self.layernorm2(x))  #with skip connection (layer normalization is applied before skip connection unlike in the paper)
    return x

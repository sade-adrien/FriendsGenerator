import parameters
import torch

#mapping functions char to int (basic tokenizer)
encode = lambda s: [parameters.stoi[c] for c in s]
decode = lambda l: ''.join([parameters.itos[i] for i in l])


def get_batch(data):
  #generate a random batch of data with input x and targets y
  ix = torch.randint(len(data) - parameters.block_size, (parameters.batch_size,))
  x = torch.stack([data[i:i+parameters.block_size] for i in ix])
  y = torch.stack([data[i+1:i+1+parameters.block_size] for i in ix])
  return x.to(parameters.device), y.to(parameters.device)
  
  
#function to give steady estimation of loss
@torch.no_grad()
def estimate_loss(model, train_data, val_data):
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(parameters.eval_iters)
    for k in range(parameters.eval_iters):
      X, Y = get_batch(train_data) if split == 'train' else get_batch(val_data)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out
import parameters
import tools
import model
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

#loading full data
with open('friends_script.txt', 'r') as file:
  text = file.read()

#data loading and train/val split
data = torch.tensor(encode(text), dtype=torch.long)
train_data = data[:int(0.9*len(data))]
val_data = data[int(0.9*len(data)):]

#Model
model = model.GPTLanguageModel().to(parameters.device)

learning_rate=1e-2
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

#training loop
for iter in tqdm(range(parameters.max_iters)):
  #update learning rate
  if iter % 5000 == 0 and iter > 0:
    learning_rate /= 10
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

  #evaluate loss
  if iter % parameters.eval_interval == 0:
    losses = tools.estimate_loss(model, train_data, val_data)
    print(f"iter {iter}: train_loss={losses['train']:.3f}, val_loss={losses['val']:.3f}")

  xb, yb = tools.get_batch(train_data)
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

final_loss = tools.estimate_loss(model, train_data, val_data)['val']

#save model
torch.save(model, f'model_{final_loss:.2f}.pth')

print(sum(p.numel() for p in model.parameters())/1e3, 'k parameters')


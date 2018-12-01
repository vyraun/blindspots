import pickle
import random
import torch

import torch.nn.functional as F

def build_data(long_data, short_data, shuffle=True):
  hiddens = []
  labels = []
  for i in range(len(short_data)-1):
    last_long = long_data[i][1][1][0][:].reshape(-1)
    last_short1 = short_data[i][1][1][0][:].reshape(-1)
    last_short2 = short_data[i+1][1][1][0][:].reshape(-1)

    inds = list(range(i-1)) + list(range(i + 2, len(short_data)))
    rand_short1 = short_data[random.choice(inds)][1][1][0][:].reshape(-1)
    rand_short2 = short_data[random.choice(inds)][1][1][0][:].reshape(-1)

    hiddens.append(torch.cat((last_long, last_short1))) #, torch.abs(last_long - last_short1), last_long * last_short1)))
    hiddens.append(torch.cat((last_long, last_short2))) #, torch.abs(last_long - last_short2), last_long * last_short2)))
    hiddens.append(torch.cat((last_long, rand_short1))) #, torch.abs(last_long - rand_short1), last_long * rand_short1)))
    hiddens.append(torch.cat((last_long, rand_short2))) #, torch.abs(last_long - rand_short2), last_long * rand_short2)))
    labels += [1,1,0,0]

  inds = list(range(len(hiddens)))
  if shuffle:
    random.shuffle(inds)
  return torch.stack(hiddens).cuda()[inds], torch.cuda.FloatTensor(labels)[inds]

# Get all the data
train_long = pickle.load(open("train.long.dumps", 'rb')) 
train_short = pickle.load(open("train.short.dumps", 'rb')) 
train_data = build_data(train_long, train_short)

test_long = pickle.load(open("test.long.dumps", 'rb')) 
test_short = pickle.load(open("test.short.dumps", 'rb')) 
test_data = build_data(test_long, test_short, shuffle=False)

class NN(torch.nn.Module):
  def __init__(
    self,
    input_size,
    hidden_size,
    num_layers,
  ):
    super(NN, self).__init__()
    assert num_layers == 1, "must be 1 for now"
    self.linear1 = torch.nn.Linear(input_size, hidden_size)
    self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
    self.linear3 = torch.nn.Linear(hidden_size, 1)
    self.dropout = torch.nn.Dropout(p=0.3)

  def forward(self, X):
    return F.sigmoid(self.linear3(self.dropout(F.tanh(self.linear2(self.dropout(F.tanh(self.linear1(X))))))))
    #return F.sigmoid(self.linear1(X))

def evaluate(model, data):
  model.eval()
  X,y = data
  y_pred = model(X)
  preds = ((y_pred > 0.5).squeeze().float() == y).float()
  acc = torch.mean(preds)
  print("Accuracy: {0}".format(acc.item()))
  model.train()

def train(num_epochs=100, batch_size=64):
  # Instantiate model, loss and optim
  model = NN(input_size=train_data[0].size(1), hidden_size=1024, num_layers=1).cuda()
  criterion = torch.nn.BCELoss()
  optim = torch.optim.Adam(model.parameters(), lr=1e-3)

  # Train
  evaluate(model, test_data)
  train_X, train_y = train_data
  for epoch in range(num_epochs):
    # Shuffle training data
    inds = list(range(len(train_X)))
    random.shuffle(inds)

    cum_loss = 0
    for i in range(len(train_X)//batch_size + 1):
      if i > 0 and i % 10 == 0:
        print("Epoch {0}, Step {1}, Loss {2}".format(epoch, i, cum_loss/i))

      # Rest optimizer
      optim.zero_grad()

      # Get batch
      batch_inds = inds[i*batch_size:(i+1)*batch_size]
      X_batch = train_X[batch_inds]
      y_batch = train_y[batch_inds]

      # Forward pass
      y_pred = model(X_batch)

      # Calculate loss and backward pass
      loss = criterion(y_pred, y_batch)
      loss.backward()
      cum_loss += loss.item()

      # Clip gradients
      torch.nn.utils.clip_grad_norm(model.parameters(), 10)

      # Step
      optim.step()

    if epoch % 5 == 0:
      evaluate(model, train_data)
      evaluate(model, test_data)

  evaluate(model, train_data)
  evaluate(model, test_data)
  evaluate(model, (test_data[0][1::4], test_data[1][1::4]))   
  evaluate(model, (test_data[0][0::4], test_data[1][0::4]))   
  torch.save(model, "SUBSTRING_EXP_MODEL_FINAL")
  import pdb; pdb.set_trace()

if __name__ == '__main__':
  seed = 42
  torch.backends.cudnn.deterministic=True
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  random.seed(seed)

  train()


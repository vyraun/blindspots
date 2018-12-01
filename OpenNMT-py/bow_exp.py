import pickle
import random
import torch

import torch.nn.functional as F

def build_vocab(data):
  all_words = [word for row in data for word in row[0]]
  vocab = ["<UNK>"] + list(set(all_words))
  return vocab

def build_data(data, shuffle=True, w2i=None):
  hiddens = []
  labels = []
  for row in data:
    for i in range(len(row[0])):
      hiddens.append(row[1][0][i])
      bow = torch.zeros(len(w2i)).cuda()
      bow[[w2i.get(w, 0) for w in row[0][:i+1]]] = 1
      labels.append(bow)
  inds = list(range(len(hiddens)))
  if shuffle:
    random.shuffle(inds)
  return torch.stack(hiddens).cuda()[inds], torch.stack(labels)[inds]

def build_data_forget(long_data, short_data, long_exp=True, w2i=None):
  hiddens = []
  labels = []
  for long_row,short_row in zip(long_data, short_data):
    if long_exp:
      row = long_row
    else:
      row = short_row

    i = len(short_row[0]) - 1
    hiddens.append(row[1][0][ len(row[0]) - 1])
    bow = torch.zeros(len(w2i)).cuda()
    bow[[w2i.get(w, 0) for w in row[0][:i+1]]] = 1
    labels.append(bow)
  inds = list(range(len(hiddens)))
  return torch.stack(hiddens).cuda()[inds], torch.stack(labels)[inds]

# Load data
train_short = pickle.load(open("train.short.dumps", 'rb')) 
test_long = pickle.load(open("test.long.dumps", 'rb')) 
test_short = pickle.load(open("test.short.dumps", 'rb')) 

# Build vocab
vocab = build_vocab(train_short)
w2i = {w:i for i,w in enumerate(vocab)}

# Build data
train_data = build_data(train_short, w2i=w2i)
test_data_long = build_data(test_long, shuffle=False, w2i=w2i)
test_data_short = build_data(test_short, shuffle=False, w2i=w2i)

test_data_forget_long = build_data_forget(test_long, test_short, long_exp=True, w2i=w2i)
test_data_forget_short = build_data_forget(test_long, test_short, long_exp=False, w2i=w2i)

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
    self.linear3 = torch.nn.Linear(hidden_size, len(w2i))
    self.dropout = torch.nn.Dropout(p=0.5)

  def forward(self, X):
    out = self.linear3(self.dropout(F.tanh(self.linear2(self.dropout(F.tanh(self.linear1(X)))))))
    return out

def evaluate(model, data):
  model.eval()
  X,y = data
  y_pred = model(X)
  loss = torch.nn.BCEWithLogitsLoss()(y_pred, y)
  print("Test loss: {0}".format(loss.item()))
  
  pred = ((F.sigmoid(y_pred) > 0.5).float() == 1)
  correct = (y == 1) 

  # TP / (TP + FP)
  prec = ( ( torch.sum(pred & correct, dim=1)/torch.sum(pred == 1 , dim=1)).float()  * (torch.sum(pred==1, dim=1)>0).float() ).float().mean()
  print("Test precision: {0}".format(prec.item()))

  # TP / (TP + FN)
  recall = ( torch.sum(pred & correct, dim=1)/torch.sum(correct == 1, dim=1) ).float().mean()
  print("Test recall: {0}".format(recall.item()))

  model.train()

def train(num_epochs=100, batch_size=64):
  # Instantiate model, loss and optim
  model = NN(input_size=train_data[0].size(1), hidden_size=1024, num_layers=1).cuda()
  criterion = torch.nn.BCEWithLogitsLoss()
  optim = torch.optim.Adam(model.parameters(), lr=1e-4)

  # Train
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

      loss = F.binary_cross_entropy_with_logits(y_pred, y_batch, y_batch * 0.8 + 0.2 *  (y_batch == 0).float())
      loss.backward()
      cum_loss += loss.item()

      # Clip gradients
      torch.nn.utils.clip_grad_norm(model.parameters(), 10)

      # Step
      optim.step()

    if epoch % 5 == 0:
      #evaluate(model, train_data)
      evaluate(model, test_data_short)

  #evaluate(model, train_data)
  evaluate(model, test_data_short)
  evaluate(model, test_data_long)
  evaluate(model, test_data_forget_long)
  evaluate(model, test_data_forget_short)
  torch.save(model, "BOW_EXP_MODEL_FINAL")
  import pdb; pdb.set_trace()

if __name__ == '__main__':
  seed = 42
  torch.backends.cudnn.deterministic=True
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  random.seed(seed)

  train()


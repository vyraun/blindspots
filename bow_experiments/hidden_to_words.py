import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pickle

from data_stuff import read_data
from data_stuff import pad_sequences

from store import log_loss, log_hyperparameter, log_model, log_exp

class vocab(object):
    def __init__(self, data):
        self.data = data
        self.word2index = {}
        self.index2word = {}
        self.number = 0

    def prep(self):
        for y in self.data:
            for x in y:
                if x not in self.word2index:
                    self.word2index[x] = self.number
                    self.index2word[self.number] = x
                    self.number = self.number + 1

class BOW(nn.Module):
    def __init__(self, hidden_size, input_vocab_size):
        super(BOW, self).__init__()
        self.linear1 = nn.Linear(hidden_size, input_vocab_size)
        self.linear2 = nn.Linear(input_vocab_size, 1)
        
    def forward(self, x):
        return self.linear2(self.linear1(x))

if __name__=='__main__':

    bow_data = pickle.load(open("predclean.txt.dumps", "rb"))
    
    input = []
    x_train = []

    for i in range(len(bow_data)):
        input.append(pad_sequences([list(bow_data[i][0])], size=bow_data[i][1][0].shape[0])[0])
        x_train.append(bow_data[i][1][0])

    input_vocab = vocab(input)
    input_vocab.prep()

    y_train = []

    # for x in input:
    #     s = [0]*input_vocab.number
    #     for w in x:
    #         s[input_vocab.word2index[w]] = 1
    #     y_train.append(s)
    
    for x in input:
        s = [0]*len(x)
        for i in range(len(x)):
            s[i] = float(input_vocab.word2index[x[i]])
        y_train.append(s)

    model = BOW(500, input_vocab.number)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    for e in range(200):
        epoch_loss = 0
        for x, y in zip(x_train, y_train):
            x = torch.tensor(x).unsqueeze(dim=0)
            y = torch.tensor(y).unsqueeze(dim=0)
            loss = criterion(model(x).squeeze(dim=2), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        print("Epoch {} Loss {}".format(e, loss))
    
    predictions = []

    # from sklearn.metrics import accuracy_score
    
    # accuracy = 0

    # for x, y in zip(x_train, y_train):
    #     x = torch.tensor(x).unsqueeze(dim=0)
    #     y = torch.tensor(y).unsqueeze(dim=0) 
    #     y_pred = model(x).squeeze(dim=2)
          
    #     accuracy += accuracy_score(y.detach().numpy(), y_pred.detach().numpy().round()) # accuracy_score(y, y_pred, normalize=False)

    # print("Accuracy {}".format(accuracy))
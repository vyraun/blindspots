import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pickle
import numpy
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
            if y not in self.word2index:
                self.word2index[y] = self.number
                self.index2word[self.number] = y
                self.number = self.number + 1

class BOW(nn.Module):
    def __init__(self, hidden_size, input_vocab_size):
        super(BOW, self).__init__()
        self.linear = nn.Linear(hidden_size, input_vocab_size)
        
    def forward(self, x):
        return F.softmax(self.linear(x))

if __name__=='__main__':

    bow_data = pickle.load(open("predclean.txt.dumps", "rb"))
    
    input = []
    x_train = []

    for i in range(len(bow_data)):
        words = pad_sequences([list(bow_data[i][0])], size=bow_data[i][1][0].shape[0])[0]
        states = bow_data[i][1][0]
        for w, s in zip(words, states):
            input.append(w)
            x_train.append(s)

    input_vocab = vocab(input)
    input_vocab.prep()

    y_train = []

    # for x in input:
    #     s = [0]*input_vocab.number
    #     for w in x:
    #         s[input_vocab.word2index[w]] = 1
    #     y_train.append(s)
    
    for x in input:
        y_train.append(input_vocab.word2index[x])

    model = BOW(500, input_vocab.number)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for e in range(1):
        epoch_loss = 0
        for x, y in zip(x_train, y_train):
            x = torch.tensor(x).unsqueeze(dim=0)
            y = torch.tensor(y).unsqueeze(dim=0)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        print("Epoch {} Loss {}".format(e, loss))
    
    from sklearn.metrics import accuracy_score
    
    accuracy = 0
    labels = []
    predictions = []

    for x, y in zip(x_train, y_train):
        x = torch.tensor(x).unsqueeze(dim=0)
        labels.append(y)
        y_pred = numpy.argmax(model(x)[0].detach().numpy())
        predictions.append(y_pred)
        # if y != y_pred:
        #     print(input_vocab.index2word[y])

    accuracy = accuracy_score(labels, predictions) # accuracy_score(y, y_pred, normalize=False)

    print("Accuracy {}".format(accuracy))

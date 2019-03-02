import pickle
import sys
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from pdb import set_trace as bp
import torch
import torch.nn.functional as F

def train_dtree():
    dtree = DecisionTreeClassifier(max_depth = 1000).fit(X_train, y_train)
    dtree_predictions = dtree.predict(X_test)

    accuracy = accuracy_score(y_test, dtree_predictions)

    print("Total Daxy " + str(len(daxy_lister)))

    total_correct = 0
    for i, x in enumerate(dtree_predictions):
        if x == y_test[i]:
            total_correct += 1

    total_daxy = 0
    for each in daxy_lister:
        if dtree_predictions[i] == daxy_index:
            total_daxy += 1

class Network(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(Network, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        return self.layer1(self.relu(inputs))

def accuracy(model, input_data, input_labs):
    model.eval()

    y_pred = F.softmax(model(input_data), dim=1).argmax(dim=1)
    
    total_correct = 0
    total_daxy = 0
    inpts_daxy = 0
    for i in range(input_labs.shape[0]):
        if input_labs[i] == y_pred[i]:
            total_correct += 1
            if input_labs[i] == daxy_index:
                total_daxy += 1
        if input_labs[i] == daxy_index:
            inpts_daxy += 1
    correct = len(input_labs)

    print("Total Correct {0}, Total Daxt Correct {1}".format(total_correct / correct, total_daxy / inpts_daxy))
    print("Total inputs = " + str(correct))

def train_network():
    model = Network(500, len(label_dict))
    input_data = torch.tensor(X_train)
    input_labs = torch.tensor(y_train)
    loss = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    batch_size = 32
    inds = list(range(len(train_data)))
    cum_loss = 0
    for epoch in range(10000):

        cum_loss = 0
        inds = list(range(len(train_data)))
        random.shuffle(inds)
        for i in range(len(inds) // batch_size + 1):
            if i > 0 and i % 2 == 0:
                print("Epoch {0}, Step {1}, Loss {2}".format(epoch, i, cum_loss/i))

            optim.zero_grad()
            batch_inds = inds[i * batch_size : (i + 1) * batch_size]
            X_batch = input_data[batch_inds]
            y_batch = input_labs[batch_inds]

            output = model(X_batch)
            loss_value = loss(output, y_batch)
            loss_value.backward()
            cum_loss += loss_value.item()

            torch.nn.utils.clip_grad_norm(model.parameters(), 10)
            print(i)
            optim.step()

        accuracy(model, torch.tensor(X_test), torch.tensor(y_test))
        #    bp()
           
if __name__ == '__main__':
    data = pickle.load(open(sys.argv[1], 'rb'))

    filer = open('lt_test.train').readlines()
    filer = [each.strip() for each in filer]
    filer_dict = {w:v for v,w in enumerate(filer)}

    new_data = [[] for i in range(len(data))]
    for each in data:
        index_this = filer_dict[' '.join(each[0])]
        new_data[index_this] = each

    data = new_data
    train_data = data[:-36]
    test_data = data[-36:]

    labels = [each[0][-2] for each in data]
    indexer = 0
    label_dict = {}
    for each in labels:
        if each not in label_dict:
            label_dict[each] = indexer
            indexer += 1

    X_train = [each[1][0][-1].numpy() for each in train_data]
    y_train = [label_dict[each[0][-2]] for each in train_data]
    X_test = [each[1][0][-1].numpy() for each in test_data]
    y_test = [label_dict[each[0][-2]] for each in test_data]

    daxy_index = label_dict['daxy']
    daxy_lister = [i for i, x in enumerate(y_test) if x == daxy_index]

    train_network()

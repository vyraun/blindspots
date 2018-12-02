import numpy as np
import pickle
Glove = {}
f = open('datasets/SCAN/mt_data/train.daxy.src.w2v', 'rb')
data = pickle.load(f)

for each in data.wv.vocab:
    Glove[each] = data[each]

f.close()

print("Done.")
X_train = []
X_train_names = []
for x in Glove:
        X_train.append(Glove[x])
        X_train_names.append(x)

X_train = np.asarray(X_train)
embedding_file = open('datasets/SCAN/mt_data/train.daxy.src.w2v.formatted', 'w')

for i, x in enumerate(X_train_names):
        embedding_file.write("%s " % x)
        for t in X_train[i]:
                embedding_file.write("%f " % t)        
        embedding_file.write("\n")

embedding_file.close()

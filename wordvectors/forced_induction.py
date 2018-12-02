import pickle
import numpy as np
from pdb import set_trace as bp

def get_induced_vectors(data_file, wordvecfile):
    wordvec = pickle.load(open(wordvecfile, 'rb'))
    train_data = open(data_file).readlines()

    vector = {}
    for each in wordvec.wv.vocab:
        vector[each] = wordvec[each]
    contextual = ['i', 'm']
    contextual_2 = ['i', 'am']
    
    sim_vec_dict = {}
    for lines in train_data:
        lines = lines.strip().split()
        if len(lines) == 4:
            if lines[-2] != 'daxy':
                if lines[0:2] == contextual or lines[0:2] == contextual_2:
                    sim_vec_dict[lines[2]] = 1
    sim_vecs = []
    for each in sim_vec_dict:
        if each in vector:
            sim_vecs.append(vector[each])

    
    sim_vecs = np.asarray(sim_vecs)
    induced_vector = np.mean(sim_vecs, axis=0)
    vector['daxy'] = induced_vector
    return vector

def write_to_file(Glove, fileto):
    X_train = []
    X_train_names = []
    for x in Glove:
        X_train.append(Glove[x])
        X_train_names.append(x)

    X_train = np.asarray(X_train)
    embedding_file = open(fileto, 'w')

    for i, x in enumerate(X_train_names):
        embedding_file.write("%s " % x)
        for t in X_train[i]:
            embedding_file.write("%f " % t)
        embedding_file.write("\n")

    embedding_file.close()
     
if __name__ == '__main__':
    wordvecfile = '../datasets/SCAN/mt_data/train.daxy.src.w2v'
    data_file = '../datasets/SCAN/mt_data/train.daxy.src'
    induced = get_induced_vectors(data_file, wordvecfile)
    write_to_file(induced, wordvecfile + '.forced_induced')

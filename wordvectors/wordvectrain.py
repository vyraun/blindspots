from gensim.models import Word2Vec
import pickle
import codecs

def train_wordvec(trainfile):
    model = Word2Vec(corpus_file=trainfile, iter=10, min_count=2, size=64, window=5, sg=1, negative=10, workers=10)                                                             
    #model.build_vocab(corpus_file=trainfile)
    #model.train(corpus_file=trainfile, total_examples=len(lines), epochs=10)
    pickle.dump(model, open(trainfile + '.w2v', 'wb'))


if __name__=='__main__':
    trainfile = '../datasets/SCAN/mt_data/forwordvecfull.daxy'
    train_wordvec(trainfile)

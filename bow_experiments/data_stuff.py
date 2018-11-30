import nltk
from nltk.corpus import stopwords 

stop_words = set(stopwords.words('english'))

def read_data(filename): # '/home/vraunak/Desktop/data.txt'

    f = open(filename)

    input = []
    output = []
    i = 0

    for line in f:
        s = line.split('    ')
        if i > 50:
            break
        i = i + 1
        input.append(nltk.word_tokenize(s[0].strip()))
        output.append(nltk.word_tokenize(s[1].strip()))

    return input, output

def pad_sequences(data, size=None):
    if size is None:
        size = len(max(data, key=lambda x: len(x)))
    else:
        size = size
    for i in range(len(data)):    
        if len(data[i])<size:
            j = len(data[i])
            while j < size:
                data[i].append('<pad>')
                j = j + 1
    
    return data

if __name__ == '__main__':
    #i, o = read_data('/home/vraunak/Desktop/data_lstm.txt')
    i, o = read_data('quora_dataset.txt')
    pi = pad_sequences(i)
    po = pad_sequences(o)
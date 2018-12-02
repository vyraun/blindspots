def create_file(train_file, test_file, filename, reps):
    train_lines = open(train_file).readlines()
    test_lines = open(test_file).readlines()
    fp = open(filename, 'w')
    for each in train_lines:
        fp.write(each)
    for i in range(reps):
        for each in test_lines:
            fp.write(each)
    fp.close()


if __name__ == '__main__':
    train_file = '../datasets/SCAN/mt_data/train.daxy.src'
    test_file = '../datasets/SCAN/mt_data/test.daxy.src'
    reps = 1000
    filename = '../datasets/SCAN/mt_data/forwordvecfull.daxy'
    create_file(train_file, test_file, filename, reps)

#put this inside opennmt-py .. opennmt-py should in the home of the main repo
python preprocess.py -train_src ../datasets/SCAN/mt_data/train.daxy.src -train_tgt ../datasets/SCAN/mt_data/train.daxy.trg -valid_src ../datasets/SCAN/mt_data/valid.daxy.random.src -valid_tgt ../datasets/SCAN/mt_data/valid.daxy.random.trg -save_data ../datasets/SCAN/daxy

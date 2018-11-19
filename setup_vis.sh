cd OpenNMT-py
wget https://github.com/pytorch/fairseq/blob/e734b0fa58fcf02ded15c236289b3bd61c4cffdf/data/prepare-iwslt14.sh
bash prepare-iwslt14.sh
cd iwslt14.tokenized.de-en
wget https://s3.amazonaws.com/opennmt-models/iwslt-brnn2.s131_acc_62.71_ppl_7.74_e20.pt
cd ..
python extract_context.py -src iwslt14.tokenized.de-en/train.de -tgt iwslt14.tokenized.de-en/train.en -model iwslt14.tokenized.de-en/iwslt-brnn2.s131_acc_62.71_ppl_7.74_e20.pt -batch_size 32
mkdir s2s
cd scripts
python h5_to_faiss.py -states ../s2s/states.h5 -data decoder_out -output decoder.faiss -stepsize 100
python h5_to_faiss.py -states ../s2s/states.h5 -data encoder_out -output encoder.faiss -stepsize 100
python h5_to_faiss.py -states ../s2s/states.h5 -data cstar -output context.faiss -stepsize 100
cd ..
sed -i -e 's/../S2Splay/model_api/processing/s2s_iwslt_ende/baseline-brnn.en-de.s154_acc_61.58_ppl_7.43_e21.pt/iwslt14.tokenized.de-en/iwslt-brnn2.s131_acc_62.71_ppl_7.74_e20.pt/g' VisServer.py
python VisServer.py
cd s2s
bash create_yaml.sh

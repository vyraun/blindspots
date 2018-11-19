cd OpenNMT-py
brew install libomp
wget https://raw.githubusercontent.com/pytorch/fairseq/e734b0fa58fcf02ded15c236289b3bd61c4cffdf/data/prepare-iwslt14.sh
bash prepare-iwslt14.sh
cd iwslt14.tokenized.de-en
wget https://s3.amazonaws.com/opennmt-models/iwslt-brnn2.s131_acc_62.71_ppl_7.74_e20.pt
cd ..
sed -i -e 's/cstarset[bcounter:] = padded_c/#cstarset[bcounter:] = padded_c/g' extract_context.py
sed -i -e 's/encoderset[bcounter:] = padded_enc/#encoderset[bcounter:] = padded_enc/g' extract_context.py
sed -i -e 's/decoderset[bcounter:] = padded_dec/#decoderset[bcounter:] = padded_dec/g' extract_context.py
mkdir s2s
python extract_context.py -src iwslt14.tokenized.de-en/train.de -tgt iwslt14.tokenized.de-en/train.en -model iwslt14.tokenized.de-en/iwslt-brnn2.s131_acc_62.71_ppl_7.74_e20.pt -batch_size 32
cd ..
cd scripts
python h5_to_faiss.py -states ../s2s/states.h5 -data decoder_out -output decoder.faiss -stepsize 100
python h5_to_faiss.py -states ../s2s/states.h5 -data encoder_out -output encoder.faiss -stepsize 100
python h5_to_faiss.py -states ../s2s/states.h5 -data cstar -output context.faiss -stepsize 100
cd ..
sed -i -e 's/../S2Splay/model_api/processing/s2s_iwslt_ende/baseline-brnn.en-de.s154_acc_61.58_ppl_7.43_e21.pt/iwslt14.tokenized.de-en/iwslt-brnn2.s131_acc_62.71_ppl_7.74_e20.pt/g' VisServer.py
python VisServer.py
cd s2s
mv states.h5 train.h5
bash create_yaml.sh
mv ./* ../../s2s/

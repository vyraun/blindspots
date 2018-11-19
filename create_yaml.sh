#!/bin/bash

if [ ! -f s2s.yaml ]; then
  echo -e "\nGenerating the s2s.yaml file\n"

  # Generate the file
  cat > s2s.yaml <<EOL
model: iwslt-brnn2.s131_acc_62.71_ppl_7.74_e20.pt
dicts:
 src: src.dict
 tgt: tgt.dict
embeddings: embs.h5
train: train.h5
indexType: faiss
indices:
 decoder: decoder.faiss
 encoder: encoder.faiss
 context: context.faiss
EOL
fi

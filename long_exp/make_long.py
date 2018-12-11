import random
import string

#clean_de = open('../iwslt14.tokenized.de-en/valid.de').readlines()
#clean_en = open('../iwslt14.tokenized.de-en/valid.en').readlines()
clean_de = open('clean.de').readlines()
clean_en = open('clean.en').readlines()
long_de = []
long_en = []
long_single = []
for i in range(len(clean_de)-1):
  #if len(clean_de[i].split()) > 50:
  #  long_de.append(clean_de[i]) 
  #  long_en.append(clean_en[i])


  long_de.append(clean_de[i].strip() + ' ' + clean_de[i+1])
  long_en.append(clean_en[i].strip() + ' ' + clean_en[i+1])

open('long2.de', 'w+').writelines(long_de)
open('long2.en', 'w+').writelines(long_en)

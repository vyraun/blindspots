import random
import string

clean_de = open('clean_full.de').readlines()
long_de = []
for i in range(len(clean_de)-1):
  #if len(clean_de[i].split()) > 50:
  #  long_de.append(clean_de[i]) 
  #  long_en.append(clean_en[i])


  long_de.append(clean_de[i].strip() + ' ' + clean_de[i+1])

open('long_train.de', 'w+').writelines(long_de)

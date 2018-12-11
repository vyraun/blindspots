import random
import string

clean = open('clean.de').readlines()
dirty = []
for j,line in enumerate(clean):
  i = random.randint(0, len(line.split())-1)
  #dirty_line = line.split()
  #dirty_line[i] = line.split()[i+1]
  #dirty_line[i+1] = line.split()[i]
  #dirty_line = ''.join(ch for ch in line if ch not in string.punctuation).split()
  dirty_line = line.split()
  if dirty_line[-1] in string.punctuation:
    dirty_line[-1] = ''
  #new_line = []
  #for i in range(len(dirty_line)):
  #  new_line.append(dirty_line[i]) 
  #  if i > 0 and  i % 5 == 0:
  #    new_line.append(',')
  #dirty_line = line.split()

  #dirty_line = new_line
  dirty.append(' '.join(dirty_line) + '\n')

open('dirty.de', 'w+').writelines(dirty)

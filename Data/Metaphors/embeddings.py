import math
import sys
import csv
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder

def metaphor_elmo(infile):

  config = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
  model = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
  elmo = ElmoEmbedder(options_file=config, weight_file=model)
  elmo_emb = []
  label = []
  #infile = "VUA_seq_formatted_train.csv"

  with open(infile, "rt", encoding = "latin-1" ) as inp:
      nlines = 0
      lines = csv.reader(inp)
      for line in lines:
        if (nlines != 0):       
          sentence = line[2]
          sentence = sentence.split()
          labels = line[3]
          l = []
          for i in labels:
            if(i == '0' or i == '1'):
              l.append(i)
          ret = list(elmo.embed_sentences(sentence))
          ret = [np.average(x, axis = 1) for x in ret]
          ret = [np.average(x, axis = 0) for x in ret]
          elmo_emb.append(ret)
          print(l)
          label.append(l)
        nlines += 1
        if(nlines%100 == 0):
          print("On line", nlines)
          
  #Save the data
  np.save('meta_labels', label)
  np.save('meta_embeds', elmo_emb)
  
def extract_emb(emb_file, lab_file):
  labels = []
  embeddings = []
  labels = list(np.load(lab_file))
  embeddings = list(np.load(emb_file))
  return(embeddings, label)

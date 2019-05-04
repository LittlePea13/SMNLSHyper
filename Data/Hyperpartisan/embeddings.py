import math
import sys
import csv
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder

def extract_emb(emb_file, lab_file):
  labels = []
  embeddings = []
  labels = list(np.load(lab_file))
  embeddings = list(np.load(emb_file))
  return(embeddings, label)
  
def hyperpart_elmo(infile, bsize, maxsent):
  # maxsent = 10
  # bsize = 5
  # infile = "output.tsv"
  hyp_elmo = []
  hyp_label = []
  with open(infile, "rt", encoding="utf8") as inp:
      nlines = 0
      for line in inp:
        fields = line.split("\t")
        title = fields[5]
        label = fields[2]
        if(label == "true"):
          label = 1
        else:
          label = 0
        tmp = fields[4]
        tmp = tmp.split(" <splt> ")[:maxsent]
        sents = [title]
        sents.extend(tmp)
        sents = [i.split() for i in sents]
#         if(nlines == 5):
#           print("Here")
#           break
        ret = list(elmo.embed_sentences(sents))
        ret = [np.average(x, axis = 1) for x in ret]
        ret = [np.average(x, axis = 0) for x in ret]
        nlines += 1
        hyp_elmo.append(ret)
        hyp_label.append(label)
        
  np.save('hyp_labels', hyp_label)
  np.save('hyp_embeds', hyp_elmo)

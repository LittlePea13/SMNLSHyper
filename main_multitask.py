from model import BiLSTMEncoder,MainModel, ModelHyper, SelfAttention, Metaphor
from multi_model import multitask_model
from helper import evaluate, evaluate_train, get_metaphor_dataset, write_board, get_document_dataset, evaluate_train_hyper, evaluate_hyper, train_valid_split
import torch.nn as nn
import torch.optim as optim
import torch
import time
from datasets import DocumentDataset
import torch.utils.data as data_utils
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import sys
from tensorboardX import SummaryWriter

# creating batch iterator for hyperpartisan data

#creating batch iterator for metaphor data
train_hp_loader = get_document_dataset('gdrive/My Drive/HyperMeta/train_embeds.npy','gdrive/My Drive/HyperMeta/train_labels.npy', batch_size = 17)
val_hp_loader = get_document_dataset('gdrive/My Drive/HyperMeta/valid_labels.npy','gdrive/My Drive/HyperMeta/valid_embeds.npy', batch_size = 17)

train_meta_loader = get_metaphor_dataset('gdrive/My Drive/HyperMeta/vua_train_embeds.npy','gdrive/My Drive/HyperMeta/vua_train_labels.npy',batch_size = 32)
val_meta_loader = get_metaphor_dataset('gdrive/My Drive/HyperMeta/vua_val_embeds.npy','gdrive/My Drive/HyperMeta/vua_val_labels.npy',batch_size = 32)

import random
l1 = len(train_hp_loader)
l2 = len(train_meta_loader)

#0 for hyper and 1 for meta
coin_flips = []
for i in range(l1):
  coin_flips.append(0)

for i in range(l2):
  coin_flips.append(1)
print(len(coin_flips))



encoder_param = { 'embed_dim': 1024,
                 'hidden_dim': 300,
                 'layers': 1,
                 'dropout_lstm' : 0,
                  'dropout_input' : 0.5 
}
hyper_param = {'embed_dim':1024, 
              'hidden_dim' : 300, 
              'layers' : 1, 
              'dropout_lstm' : 0, 
              'dropout_input':0.5, 
              'dropout_FC':0.1,
              'dropout_lstm_hyper' : 0,
              'dropout_input_hyper' : 0,
              'dropout_attention' : 0,
              'num_classes' : 2,
              'learning_rate':0.001}

meta_param = {
    'hidden_dim' : 150, 
     'dropout_FC' : 0.1,
      'num_classes' : 2,
          'learning_rate': 0.001
}
'''
meta_param = {
    'hidden_dim' : 150, 
     'dropout_FC' : 0.3,
      'num_classes' : 2,
    'learning_rate': 0.001
}
'''

writer = SummaryWriter()
model = multitask_model(encoder_param, hyper_param, meta_param)
#model = multitask_soft_model(encoder_param, hyper_param, meta_param)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    model.to(device=device)
nll_criterion = nn.NLLLoss()
num_epochs = 10

val_loss = []
val_f1 = []
counter = 0
met_model_optimizer = optim.Adam(model.parameters(), lr=hyper_param['learning_rate'])

for epoch in range(num_epochs):
    
    print("Starting epoch {}".format(epoch + 1))
    random.shuffle(coin_flips)
    
    for i in coin_flips:
      if(i == 0):
        #batch to be picked up from hyperpartisan dataset
        batch = next(iter(train_hp_loader))
        #(data, doc_len, labels, sen_len)
        data = batch[0]
        doc_len = batch[1]
        labels = batch[2]
        sen_len = batch[3]
        is_doc = True
      else:
        #batch to be picked up from metaphor dataset
        batch = next(iter(train_meta_loader))
        #(data, lengths, labels)
        is_doc = False
        data = batch[0]
        lengths = batch[1]
        labels = batch[2]
      
      if torch.cuda.is_available():
        labels = labels.to(device=torch.device('cuda'))
      
      if(is_doc):
        
        meta_pred, hyp_pred = model(is_doc, data, sen_len, doc_len) 
        batch_loss = nll_criterion(hyp_pred.view(-1, 2), labels.view(-1))
        precision, recall, f1, eval_accuracy = evaluate_train_hyper(labels, hyp_pred)
        met_model_optimizer.zero_grad()
        batch_loss.backward()
        met_model_optimizer.step()
        counter += 1
        write_board(writer,'Hyper/Train', precision, recall, f1, eval_accuracy, batch_loss.item(), counter)

      else:
        
        meta_pred, hyp_pred = model(is_doc, data, lengths, 1)
        batch_loss = nll_criterion(meta_pred.view(-1, 2), labels.view(-1))
        precision, recall, f1, eval_accuracy = evaluate_train(labels, meta_pred, lengths)
      
        met_model_optimizer.zero_grad()
        batch_loss.backward()
        met_model_optimizer.step()
        counter += 1
        write_board(writer,'Meta/Train', precision, recall, f1, eval_accuracy, batch_loss.item(), counter)
      print("Iteration {}. Train Loss {}. Train Accuracy {}. Train Precision {}. Train Recall {}. Train F1 {}.".format(counter, batch_loss.item(), eval_accuracy, precision, recall, f1))
      if counter % 5 == 0:
        if is_doc:
          avg_eval_loss, precision, recall, f1, eval_accuracy = evaluate_hyper(val_hp_loader, model, nll_criterion, device)
          write_board(writer,'Hyper/Val', precision, recall, f1, eval_accuracy, avg_eval_loss, counter)
          print("Iteration {}. Hyper Validation Loss {}. Validation Accuracy {}. Validation Precision {}. Validation Recall {}. Validation F1 {}.".format(counter, avg_eval_loss, eval_accuracy, precision, recall, f1))
        else:
          avg_eval_loss, precision, recall, f1, eval_accuracy = evaluate(val_meta_loader, model, nll_criterion, device)
          write_board(writer,'Meta/Val', precision, recall, f1, eval_accuracy, avg_eval_loss, counter)
          print("Iteration {}. Meta Validation Loss {}. Validation Accuracy {}. Validation Precision {}. Validation Recall {}. Validation F1 {}.".format(counter, avg_eval_loss, eval_accuracy, precision, recall, f1))
print("First Training done!")

        

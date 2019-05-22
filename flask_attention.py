    
from Data.Metaphors.embeddings import extract_emb
from model import BiLSTMEncoder,MainModel, ModelHyper, multitask_model,multitask_soft_model
from helper import evaluate, evaluate_train, get_metaphor_dataset, write_board, get_document_dataset, evaluate_train_hyper,evaluate_hyper
import torch.nn as nn
import torch.optim as optim
import torch
from tensorboardX import SummaryWriter
import time
import argparse
import numpy as np
import torch
from datasets import SentenceDataset,TestDocumentDataset, AdaptSampler
import torch.utils.data as data_utils
import numpy as np

import argparse
#import tensorflow as tf

from flask import Flask
from flask import render_template
from flask import request
from flask import Response
from flask import jsonify
import subprocess
#from util import text_util
import sys
import ast
import csv
def test_model(model_name, multi_model=True ):
    trained_model = torch.load(model_name, map_location='cpu')
    hyper_param = trained_model['hyperparameters']
    encoder_param = trained_model['encoderparameters']
    meta_param = trained_model['metaparameters']
    
    if multi_model ==  True:
        model = multitask_soft_model(encoder_param, hyper_param, meta_param)
    else:
        model = multitask_model(encoder_param, hyper_param, meta_param)

    data = np.load('hyp_embeds.npy',allow_pickle=True)
    data = np.array([np.array(xi) for xi in data])
    dataset = TestDocumentDataset(data[0], 200)
    evaluation_dataloader = data_utils.DataLoader(dataset, batch_size=1,
                              collate_fn=TestDocumentDataset.collate_fn, shuffle=False)
    model.eval()
    all_doc_ids = range(0,len(data[0]))
    model.load_state_dict(trained_model['state_dict'])
    total_examples = 0
    total_eval_loss = 0
    predict_confidence = []
    predict_list = []
    for (eval_text, doc_len, eval_lengths) in evaluation_dataloader:
        if torch.cuda.is_available():
            doc_len = doc_len.to(device=device)
        predicted_meta, predicted = model(eval_text, eval_lengths, doc_len, True)
        _, predicted_labels = torch.max(predicted.view(-1,2).data, 1)
        predict_list.append(predicted_labels.item())
        predict_confidence.append(torch.exp(predicted.view(-1,2).data).data)
        total_examples += doc_len.size(0)
    predict_confidence = torch.cat(predict_confidence, dim=0)
    print(predict_confidence)
    print(predict_list)
    # predicted_labels = [predict for element in predicted_labels for predict in element]
    # Set the model back to train mode, which activates dropout again.

    predicted_meta = torch.exp(predicted_meta[:,:,1]).tolist()
    activation_maps = []
    activation_maps_words = []
    activation_maps_words_meta = []
    activation_maps_sen = []
    str_words = []
    with open('attention_stuff.csv','r') as csvinput:
        with open('output.csv', 'w') as csvoutput:
            with open('Att_article.txt', 'r') as textfile:
                writer = csv.writer(csvoutput, delimiter='\t', lineterminator='\n')
                reader = csv.reader(csvinput, delimiter='\t')
                text = csv.reader(textfile, delimiter=' ', lineterminator='\n')
                all = []
                for i, row in enumerate(zip(reader,text)):
                    str_sen = ''
                    new_row = []
                    #print(zip(row[0][0][:len(row[1])],row[1]))
                    #new_row.append(predicted_meta[i])
                    for word in row[1]:
                        str_sen += word + ' '
                    print(str_sen)
                    str_words.append(str_sen)
                    activation_maps_words.append(list(zip(row[1],list(map(float, ast.literal_eval(row[0][0])))[:len(row[1])])))
                    activation_maps_words_meta.append(list(zip(row[1],predicted_meta[i][:len(row[1])])))
                    #new_row.append(list(zip(row[1],predicted_meta[i][:len(row[1])], ast.literal_eval(row[0][0])[:len(row[1])])))#,predicted_meta[i])))
                    new_row.append(row[0][1])
                    new_row.append(row[1])
                    new_row.append(predicted_meta[i][:len(row[1])])
                    new_row.append(ast.literal_eval(row[0][0])[:len(row[1])])
                    activation_maps_sen.append(row[0][1])
                    all.append(new_row)
        
                writer.writerows(all)
    activation_maps = list(zip(activation_maps_words, activation_maps_sen))   
    #print(activation_maps)         
    return activation_maps, predict_list, predict_confidence, str_words, activation_maps_words_meta

print(sys.executable)
#print(str_words)
app = Flask(__name__)

#graph = tf.get_default_graph()

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/activations')
def activations():
    if request.method == 'GET':
        text = request.args.get('text', '')
        try:
            subprocess.check_call(['/anaconda3/envs/statnlp/bin/python3 Data/Hyperpartisan/attention.py -text "' + text+'"'], shell=True)
            subprocess.check_call(['/anaconda3/envs/statnlp/bin/python3 Data/elmo_power_att.py -TSV output_trial.tsv'], shell=True)
            activation_maps, predicted_labels, predict_confidence, str_words, activation_maps_words_meta = test_model('Model/multitask_model.pt', False)
            print(activation_maps_words_meta)
        except subprocess.CalledProcessError:
            print('didnt work')
            pass # error
        except OSError:
            print('didnt work')

        if len(text.strip()) == 0:
            return Response(status=400)
        data = {
            'activations': activation_maps,
            'activations_metaphos': activation_maps_words_meta,
            'normalizedText': str_words,
            'prediction': predicted_labels,
            'binary': True
        }
        return jsonify(data)
    else:
        return Response(status=501)
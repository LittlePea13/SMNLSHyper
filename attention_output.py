    
from Data.Metaphors.embeddings import extract_emb
from model import BiLSTMEncoder,MainModel, ModelHyper, multitask_model,multitask_soft_model,HyperModel1
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

def test_model(args):
    trained_model = torch.load(args.model, map_location='cpu')
    if args.multi_model == '0':
        hyper_param = trained_model['hyperparameters']
        encoder_param = trained_model['encoderparameters']
        meta_param = trained_model['metaparameters']
        model = multitask_soft_model(encoder_param, hyper_param, meta_param)
    elif args.multi_model == '1':
        hyper_param = trained_model['hyperparameters']
        encoder_param = trained_model['encoderparameters']
        meta_param = trained_model['metaparameters']
        model = multitask_model(encoder_param, hyper_param, meta_param)
    elif args.multi_model == '2':
        hyper_param = trained_model['hyperparameters']
        model = HyperModel1(1024, hyper_param['hidden_dim'],1, hyper_param['dropout_lstm'], hyper_param['dropout_input'], hyper_param['dropout_FC'], hyper_param['dropout_lstm_hyper'], hyper_param['dropout_input_hyper'], hyper_param['dropout_attention'], 2)
        '''        model = HyperModel1(embed_dim=1024, 
                    hidden_dim = hyperparameters['hidden_dim'], 
                    layers = 1,
                    dropout_lstm = hyperparameters['dropout_lstm'], 
                    dropout_input=hyperparameters['dropout_input'], 
                    dropout_FC=hyperparameters['dropout_FC'],
                    dropout_lstm_hyper = hyperparameters['dropout_lstm_hyper'],
                    dropout_input_hyper = hyperparameters['dropout_input_hyper'],
                    dropout_attention = hyperparameters['dropout_attention'],
                    num_classes = 2)'''
    data = np.load('Data/Hyperpartisan/valid_embeds.npy',allow_pickle=True)
    data_labels = np.load('Data/Hyperpartisan/valid_labels.npy',allow_pickle=True)
    data = np.array([np.array(xi) for xi in data])
    dataset = TestDocumentDataset(data, 200)
    evaluation_dataloader = data_utils.DataLoader(dataset, batch_size=8,
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
        if args.multi_model == '2':
            predicted = model(eval_text, eval_lengths, doc_len)
            predicted_meta = None
        else:
            predicted_meta, predicted = model(eval_text, eval_lengths, doc_len, True)
            print(predicted_meta.shape)
        _, predicted_labels = torch.max(predicted.view(-1,2).data, 1)
        predict_list.append(predicted_labels)
        predict_confidence.append(torch.exp(predicted.view(-1,2).data).data)
        total_examples += doc_len.size(0)
    predict_confidence = torch.cat(predict_confidence, dim=0)
    # predicted_labels = [predict for element in predicted_labels for predict in element]
    # Set the model back to train mode, which activates dropout again.
    print('Number of predictions ',len(predict_list))

    print(predict_confidence.shape, 'results')
    import ast
    import csv
    predicted_meta = torch.exp(predicted_meta[:,:,1]).tolist()
    activation_maps = []
    activation_maps_words = []
    activation_maps_sen = []
    with open('attention_stuff.csv','r') as csvinput:
        with open('output.csv', 'w') as csvoutput:
            with open('Att_article.txt', 'r') as textfile:
                writer = csv.writer(csvoutput, delimiter='\t', lineterminator='\n')
                reader = csv.reader(csvinput, delimiter='\t')
                text = csv.reader(textfile, delimiter=' ', lineterminator='\n')
                all = []
                for i, row in enumerate(zip(reader,text)):
                    new_row = []
                    #print(zip(row[0][0][:len(row[1])],row[1]))
                    #new_row.append(predicted_meta[i])
                    activation_maps_words.append(list(zip(row[1], ast.literal_eval(row[0][0])[:len(row[1])])))
                    new_row.append(list(zip(row[1], ast.literal_eval(row[0][0])[:len(row[1])])))#,predicted_meta[i])))
                    new_row.append(row[0][1])
                    activation_maps_sen.append(row[0][1])
                    all.append(new_row)
        
                writer.writerows(all)
    activation_maps = list(zip(activation_maps_words, activation_maps_sen))            
    return activation_maps 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", default='hyper.pt', type=str, required=True, help="Article XML file")
    parser.add_argument("-multi_model", default='0', type=str, required=False, help="Article XML file")
    parser.add_argument("-out", type=str, help="Output (tsv) file")
    args = parser.parse_args()
    test_model(args)
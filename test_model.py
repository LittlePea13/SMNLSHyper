    
from Data.Metaphors.embeddings import extract_emb
from model import BiLSTMEncoder,MainModel, ModelHyper, HyperModel1
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
    hyperparameters = trained_model['hyperparameters']
    model = HyperModel1(embed_dim=1024, 
                    hidden_dim = hyperparameters['hidden_dim'], 
                    layers = 1,
                    dropout_lstm = hyperparameters['dropout_lstm'], 
                    dropout_input=hyperparameters['dropout_input'], 
                    dropout_FC=hyperparameters['dropout_FC'],
                    dropout_lstm_hyper = hyperparameters['dropout_lstm_hyper'],
                    dropout_input_hyper = hyperparameters['dropout_input_hyper'],
                    dropout_attention = hyperparameters['dropout_attention'],
                    num_classes = 2)

    data = np.load('hyp_embeds.npy',allow_pickle=True)
    data = np.array([np.array(xi) for xi in data])
    dataset = TestDocumentDataset(data[0], 200)
    evaluation_dataloader = data_utils.DataLoader(dataset, batch_size=20,
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
        predicted = model(eval_text, eval_lengths, doc_len)
        _, predicted_labels = torch.max(predicted.data, 1)
        predict_list.append(predicted_labels)
        predict_confidence.append(torch.exp(predicted.data).data)
        total_examples += doc_len.size(0)
    predict_confidence = torch.cat(predict_confidence, dim=0)
    # predicted_labels = [predict for element in predicted_labels for predict in element]
    # Set the model back to train mode, which activates dropout again.
    print('Number of predictions ',len(predict_list))
    print(predict_confidence.shape, 'results')
    all_pred = toEvaluationFormat(all_doc_ids, predict_confidence)
    with open(args.out, 'w') as fo:
        for item in all_pred:
            fo.write(item)

def toEvaluationFormat(all_doc_ids, all_prediction):
    evaluationFormatList = []
    for i in range(len(all_doc_ids)):
        current_doc_id = all_doc_ids[i]
        current_prob = all_prediction[i][0]
        #current_prob = all_prediction[i]
        print(current_prob)
        if current_prob > 0.5:
            current_pred = 'false'
        else:
            current_prob = 1 - current_prob
            current_pred = 'true'
        evaluationFormat = str(current_doc_id).zfill(7) + ' ' + str(current_pred) + ' ' + str(current_prob) + '\n'
        evaluationFormatList.append(evaluationFormat)
    return evaluationFormatList

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", default='hyper.pt', type=str, required=True, help="Article XML file")
    parser.add_argument("-out", type=str, help="Output (tsv) file")
    args = parser.parse_args()
    test_model(args)
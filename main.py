import ast
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datasets import SentenceDataset,DocumentDataset
from Data.Metaphors.embeddings import extract_emb
import torch.utils.data as data_utils
from model import BiLSTMEncoder,MainModel
from helper import evaluate, evaluate_train
import torch.nn as nn
import torch.optim as optim
import torch
from tensorboardX import SummaryWriter

def load_elmo_dataset(path, max_len=200):
    '''
    load ELMo embedding from tsv file.
    :param path: tsv file path.
    :param to_pickle: Convert elmo embeddings to .npy file, avoid read and pad every time.
    :return: elmo embedding and its label.
    '''
    X = []
    label = []
    ids = []
    i = 0
    with open(path, 'rb') as inf:
        for line in inf:
            gzip_fields = line.decode('utf-8').split('\t')
            gzip_id = gzip_fields[0]
            gzip_label = gzip_fields[1]
            elmo_embd_str = gzip_fields[4].strip()
            elmo_embd_list = ast.literal_eval(elmo_embd_str)
            elmo_embd_array = np.array(elmo_embd_list)
            padded_seq = sequence.pad_sequences([elmo_embd_array], maxlen=max_len, dtype='float32')[0]
            X.append(padded_seq)
            label.append(gzip_label)
            ids.append(gzip_id)
            i += 1
            print(i)
    # transform label to variable
    Y = l_encoder.fit_transform(label)
    #tensor_data = torch.from_numpy(np.array(X))
    #tensor_target = torch.from_numpy(np.array(Y))
    return np.array(X), np.array(Y)

if __name__ == "__main__":

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    batch_size = 64
    hidden_dim = 300
    emb_file = 'Data/Metaphors/VUA/vua_train_embeds.npy'
    lab_file = 'Data/Metaphors/VUA/vua_train_labels.npy'
    emb_file_val = 'Data/Metaphors/VUA/vua_val_embeds.npy'
    lab_file_val = 'Data/Metaphors/VUA/vua_val_labels.npy'

    data, labels = extract_emb(emb_file, lab_file)
    data_val, labels_val = extract_emb(emb_file_val, lab_file_val)

    dataset = SentenceDataset(data, labels, 200)
    dataset_val = SentenceDataset(data_val, labels_val, 200)

    loader_dataset = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=SentenceDataset.collate_fn)
    val_loader = data_utils.DataLoader(dataset_val, batch_size=batch_size, shuffle=True,
                                  collate_fn=SentenceDataset.collate_fn)

    model = MainModel(embed_dim=1024, hidden_dim = hidden_dim, layers = 1, dropout_lstm = 0, dropout_input=0.5, dropout_FC=0.1, num_classes = 2)
    if torch.cuda.is_available():
        model.to(device=device)
    nll_criterion = nn.NLLLoss()
    # Set up an optimizer for updating the parameters of the rnn_clf
    met_model_optimizer = optim.Adam(model.parameters(), lr=0.005)
    # Number of epochs (passes through the dataset) to train the model for.
    num_epochs = 20

    val_loss = []
    val_f1 = []
    counter = 0

    writer = SummaryWriter()

    for epoch in range(num_epochs):
        print("Starting epoch {}".format(epoch + 1))
        for (data, lengths, labels) in loader_dataset:
            if torch.cuda.is_available():
                data.to(device=device)
                lengths.to(device=device)
                labels.to(device=device)
            predicted = model(data, lengths)
            batch_loss = nll_criterion(predicted.view(-1, 2), labels.view(-1))
            precision, recall, f1, eval_accuracy = evaluate_train(labels, predicted, lengths)
            met_model_optimizer.zero_grad()
            batch_loss.backward()
            met_model_optimizer.step()
            counter += 1
            writer.add_scalar('Train/F1', f1, (counter))
            writer.add_scalar('Train/precision', precision, (counter))
            writer.add_scalar('Train/recall', recall, (counter))
            writer.add_scalar('Train/accuracy', recall, (counter))
            writer.add_scalar('Train/Loss', batch_loss.item(), (counter))
            if counter % 2 == 0:
                avg_eval_loss, precision, recall, f1, eval_accuracy = evaluate(val_loader, model, nll_criterion, device)
                writer.add_scalar('Val/F1', f1, (counter))
                writer.add_scalar('Val/precision', precision, (counter))
                writer.add_scalar('Val/recall', recall, (counter))
                writer.add_scalar('Val/accuracy', eval_accuracy, (counter))
                writer.add_scalar('Val/Loss', avg_eval_loss, (counter))
                #print(avg_eval_loss)
                #val_loss.append(avg_eval_loss)
                #val_f1.append(f1)
                print("Iteration {}. Validation Loss {}. Validation Accuracy {}. Validation Precision {}. Validation Recall {}. Validation F1 {}.".format(counter, avg_eval_loss, eval_accuracy, precision, recall, f1))
    print("Training done!")
    writer.close()
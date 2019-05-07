import ast
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datasets import SentenceDataset,DocumentDataset
from Data.Metaphors.embeddings import extract_emb
import torch.utils.data as data_utils
from model import BiLSTMEncoder,MainModel
from helper import evaluate
import torch.nn as nn
import torch.optim as optim

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
    batch_size = 64
    hidden_dim = 512
    emb_file = 'Data/Metaphors/meta_embeds.npy'
    lab_file = 'Data/Metaphors/meta_labels.npy'
    data, labels = extract_emb(emb_file, lab_file)
    #data, labels = load_elmo_dataset(path,200)
    dataset = SentenceDataset(data, labels, 200)
    loader_doc = DocumentDataset(data, labels, 200)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = data_utils.random_split(dataset, [train_size, test_size])

    loader_dataset = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=SentenceDataset.collate_fn)
    val_dataset = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=SentenceDataset.collate_fn)

    model = MainModel(embed_dim=1024, hidden_dim = hidden_dim, layers = 1, dropout_lstm = 0, dropout_input=0, dropout_FC=0, num_classes = 2)

    nll_criterion = nn.NLLLoss()
    # Set up an optimizer for updating the parameters of the rnn_clf
    met_model_optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
    # Number of epochs (passes through the dataset) to train the model for.
    num_epochs = 20

    val_loss = []
    val_f1 = []
    counter = 0

    for epoch in range(num_epochs):
        print("Starting epoch {}".format(epoch + 1))
        for (data, lengths, labels) in loader_dataset:
            predicted = model(data, lengths)
            batch_loss = nll_criterion(predicted.view(-1, 2), labels.view(-1))
            print(batch_loss.item())
            met_model_optimizer.zero_grad()
            batch_loss.backward()
            met_model_optimizer.step()
            counter += 1
            if counter % 2 == 0:
                avg_eval_loss, precision, recall, f1, eval_accuracy = evaluate(val_dataset, model, nll_criterion)
                #print(avg_eval_loss)
                #val_loss.append(avg_eval_loss)
                #val_f1.append(f1)
                print("Iteration {}. Validation Loss {}. Validation Accuracy {}. Validation Precision {}. Validation Recall {}. Validation F1 {}.".format(counter, avg_eval_loss, eval_accuracy, precision, recall, f1))
    print("Training done!")
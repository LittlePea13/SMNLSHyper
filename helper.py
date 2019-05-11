import numpy as np
import torch
from datasets import SentenceDataset,DocumentDataset
import torch.utils.data as data_utils

def evaluate(evaluation_dataloader, model, criterion, device):
    model.eval()
    total_examples = 0
    total_eval_loss = 0
    confusion_matrix = np.zeros((2, 2))
    for (eval_text, eval_lengths, eval_labels) in evaluation_dataloader:
        if torch.cuda.is_available():
            eval_text.to(device=device)
            eval_lengths.to(device=device)
            eval_labels.to(device=device)
        predicted = model(eval_text, eval_lengths)
        total_eval_loss += criterion(predicted.view(-1, 2), eval_labels.view(-1))
        _, predicted_labels = torch.max(predicted.data, 2)
        total_examples += eval_lengths.size(0)
        confusion_matrix = update_confusion_matrix(confusion_matrix, predicted_labels, eval_labels.data, eval_lengths)
    average_eval_loss = total_eval_loss / evaluation_dataloader.__len__()
    precision = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[1])
    recall = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[:, 1])
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = 100 * (confusion_matrix[1, 1] + confusion_matrix[0, 0]) / np.sum(confusion_matrix)
    # Set the model back to train mode, which activates dropout again.
    model.train()
    print('Number of predictions ',confusion_matrix.sum())
    return average_eval_loss.data.item(), precision, recall, f1, accuracy

def evaluate_hyper(evaluation_dataloader, model, criterion, device):
    model.eval()
    total_examples = 0
    total_eval_loss = 0
    confusion_matrix = np.zeros((2, 2))
    for (eval_text, doc_len, eval_labels, eval_lengths) in evaluation_dataloader:
        if torch.cuda.is_available():
            doc_len = doc_len.to(device=device)
            eval_labels = eval_labels.to(device=device)
        predicted = model(eval_text, eval_lengths, doc_len)
        total_eval_loss += criterion(predicted.view(-1, 2), eval_labels.view(-1))
        _, predicted_labels = torch.max(predicted.data, 1)
        total_examples += doc_len.size(0)
        confusion_matrix = update_hyper_confusion_matrix(confusion_matrix, predicted_labels, eval_labels.data)
    average_eval_loss = total_eval_loss / evaluation_dataloader.__len__()
    precision = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[1])
    recall = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[:, 1])
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = 100 * (confusion_matrix[1, 1] + confusion_matrix[0, 0]) / np.sum(confusion_matrix)
    # Set the model back to train mode, which activates dropout again.
    model.train()
    print('Number of predictions ',confusion_matrix.sum())
    return average_eval_loss.data.item(), precision, recall, f1, accuracy

def evaluate_train(labels, predictions, lengths):
    _, predicted_labels = torch.max(predictions.data, 2)
    confusion_matrix = np.zeros((2, 2))
    confusion_matrix = update_confusion_matrix(confusion_matrix, predicted_labels, labels.data, lengths)
    precision = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[1])
    recall = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[:, 1])
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = 100 * (confusion_matrix[1, 1] + confusion_matrix[0, 0]) / np.sum(confusion_matrix)
    return precision, recall, f1, accuracy

def evaluate_train_hyper(labels, predictions):
    _, predicted_labels = torch.max(predictions.data, 1)
    print(predicted_labels)
    print(labels)
    confusion_matrix = np.zeros((2, 2))
    confusion_matrix = update_hyper_confusion_matrix(confusion_matrix, predicted_labels, labels.data)
    precision = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[1])
    recall = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[:, 1])
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = 100 * (confusion_matrix[1, 1] + confusion_matrix[0, 0]) / np.sum(confusion_matrix)
    return precision, recall, f1, accuracy

def update_confusion_matrix(matrix, predictions, labels, sen_len):
    for i in range(len(sen_len)):
        prediction = predictions[i]
        label = labels[i]
        sentence = sen_len[i]
        for j in range(sentence):
            p = prediction[j]
            l = label[j]
            matrix[p][l] += 1
    return matrix

def update_hyper_confusion_matrix(matrix, predictions, labels):
    for j in range(len(labels)):
        p = predictions[j]
        l = labels[j]
        matrix[p][l] += 1
    return matrix

def get_metaphor_dataset(filename_data, filename_labels, batch_size):
    data, labels = extract_emb(filename_data, filename_labels)
    dataset = SentenceDataset(data, labels, 200)
    return data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=SentenceDataset.collate_fn)

def get_document_dataset(filename_data, filename_labels, batch_size):
    data, labels = extract_emb(filename_data, filename_labels)
    dataset = DocumentDataset(data, labels, 200)
    train_data, valid_data = train_valid_split(dataset, split_fold=8)
    train_loader = data_utils.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                  collate_fn=DocumentDataset.collate_fn)
    val_loader = data_utils.DataLoader(valid_data, batch_size=batch_size, shuffle=True,
                                  collate_fn=DocumentDataset.collate_fn)
    return train_loader, val_loader

def write_board(writer, partition, precision, recall, f1, accuracy, loss, step):
    writer.add_scalar(partition + '/F1', f1, (step))
    writer.add_scalar(partition + '/precision', precision, (step))
    writer.add_scalar(partition + '/recall', recall, (step))
    writer.add_scalar(partition + '/accuracy', accuracy, (step))
    writer.add_scalar(partition + '/Loss', loss, (step))

def train_valid_split(ds, split_fold=10, random_seed=None):
    '''
    This is a pytorch generic function that takes a data.Dataset object and splits it to validation and training
    efficiently.
    :return:
    '''
    if random_seed!=None:
        np.random.seed(random_seed)
    dslen=len(ds)
    indices= list(range(dslen))
    valid_size=dslen//split_fold
    np.random.shuffle(indices)
    train_mapping=indices[valid_size:]
    valid_mapping=indices[:valid_size]
    train=GenHelper(ds, dslen - valid_size, train_mapping)
    valid=GenHelper(ds, valid_size, valid_mapping)
    return train, valid

class GenHelper(data_utils.Dataset):
    def __init__(self, mother, length, mapping):
        # here is a mapping from this index to the mother ds index
        self.mapping=mapping
        self.length=length
        self.mother=mother
    def __getitem__(self, index):
        return self.mother[self.mapping[index]]
    def __len__(self):
        return self.length

def extract_emb(emb_file, lab_file):
  labels = []
  embeddings = []
  labels = np.load(lab_file,allow_pickle=True)
  embeddings = np.load(emb_file,allow_pickle=True)
  labels=np.array([np.array(xi).astype(np.int) for xi in labels])
  embeddings=np.array([np.array(xi) for xi in embeddings])
  return embeddings, labels
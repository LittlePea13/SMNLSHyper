import numpy as np
import torch

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

def evaluate_train(labels, predictions, lengths):
    _, predicted_labels = torch.max(predictions.data, 2)
    confusion_matrix = np.zeros((2, 2))
    confusion_matrix = update_confusion_matrix(confusion_matrix, predicted_labels, labels.data, lengths)
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
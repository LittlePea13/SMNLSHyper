import numpy as np
import torch
def evaluate(evaluation_dataloader, model, criterion):
    
    model.eval()

    # total_examples = total number of words
    total_examples = 0
    total_eval_loss = 0
    #     confusion_matrix = np.zeros((len(idx2pos), 2, 2))
    for (eval_text, eval_lengths, eval_labels) in evaluation_dataloader:
   
        # predicted shape: (batch_size, seq_len, 2)
        predicted = model(eval_text, eval_lengths)
        # Calculate loss for this test batch. This is averaged, so multiply
        # by the number of examples in batch to get a total.
        total_eval_loss += criterion(predicted.view(-1, 2), eval_labels.view(-1))
        # get 0 or 1 predictions
        # predicted_labels: (batch_size, seq_len)
        _, predicted_labels = torch.max(predicted.data, 2)
        total_examples += eval_lengths.size(0)
    #         confusion_matrix = update_confusion_matrix(confusion_matrix, predicted_labels, eval_labels.data, eval_pos_seqs)

    average_eval_loss = total_eval_loss / evaluation_dataloader.__len__()

    # Set the model back to train mode, which activates dropout again.
    model.train()
    return average_eval_loss.data.item()